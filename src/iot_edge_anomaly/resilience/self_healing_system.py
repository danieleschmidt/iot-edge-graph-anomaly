"""
Advanced Self-healing and Auto-recovery System for IoT Edge Anomaly Detection.

This module provides comprehensive self-healing capabilities with:
- Predictive failure detection using ML algorithms
- Automated recovery with rollback mechanisms
- Cascade failure prevention across distributed components
- Automatic model retraining triggers on drift detection
- Resource exhaustion prevention with auto-scaling
- Performance degradation early warning systems
"""

import asyncio
import logging
import pickle
import time
import json
import os
from typing import Dict, Any, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import uuid
import torch
import numpy as np
from collections import defaultdict, deque
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor
import joblib

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """System health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"


class RecoveryAction(Enum):
    """Types of recovery actions."""
    RESTART_COMPONENT = "restart_component"
    ROLLBACK_MODEL = "rollback_model"
    SCALE_RESOURCES = "scale_resources"
    SWITCH_TO_BACKUP = "switch_to_backup"
    RETRAIN_MODEL = "retrain_model"
    REDUCE_LOAD = "reduce_load"
    EMERGENCY_SHUTDOWN = "emergency_shutdown"


class FailurePrediction(Enum):
    """Failure prediction confidence levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    IMMINENT = "imminent"


@dataclass
class HealthMetric:
    """Represents a health metric."""
    name: str
    value: float
    threshold_warning: float
    threshold_critical: float
    timestamp: datetime = field(default_factory=datetime.now)
    trend: Optional[str] = None  # "increasing", "decreasing", "stable"
    prediction: Optional[float] = None  # Predicted future value
    
    def get_status(self) -> HealthStatus:
        """Get status based on thresholds."""
        if self.value >= self.threshold_critical:
            return HealthStatus.CRITICAL
        elif self.value >= self.threshold_warning:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "value": self.value,
            "threshold_warning": self.threshold_warning,
            "threshold_critical": self.threshold_critical,
            "timestamp": self.timestamp.isoformat(),
            "status": self.get_status().value,
            "trend": self.trend,
            "prediction": self.prediction
        }


@dataclass
class RecoveryPlan:
    """Represents a recovery plan for a specific failure scenario."""
    plan_id: str
    trigger_condition: str
    recovery_actions: List[Dict[str, Any]]
    estimated_recovery_time: float
    success_criteria: Dict[str, Any]
    rollback_plan: Optional[List[Dict[str, Any]]] = None
    priority: int = 1  # Lower number = higher priority
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "plan_id": self.plan_id,
            "trigger_condition": self.trigger_condition,
            "recovery_actions": self.recovery_actions,
            "estimated_recovery_time": self.estimated_recovery_time,
            "success_criteria": self.success_criteria,
            "rollback_plan": self.rollback_plan,
            "priority": self.priority
        }


class PredictiveFailureDetector:
    """
    ML-based predictive failure detection system.
    
    Features:
    - Time series analysis for trend prediction
    - Anomaly detection in system metrics
    - Pattern recognition for failure precursors
    - Multi-variate analysis across system components
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize predictive failure detector."""
        self.config = config or {}
        
        # ML models for prediction
        self.failure_prediction_models: Dict[str, Any] = {}
        
        # Historical data storage
        self.metric_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.failure_history: List[Dict[str, Any]] = []
        
        # Prediction thresholds
        self.prediction_thresholds = {
            FailurePrediction.LOW: 0.3,
            FailurePrediction.MEDIUM: 0.5,
            FailurePrediction.HIGH: 0.7,
            FailurePrediction.IMMINENT: 0.9
        }
        
        # Initialize models
        self._initialize_prediction_models()
    
    def _initialize_prediction_models(self):
        """Initialize ML models for failure prediction."""
        # Simple time series predictor
        class TimeSeriesPredictor:
            def __init__(self):
                self.model = None
                self.scaler = None
                self.lookback_window = 20
                
            def fit(self, data: np.ndarray):
                """Fit the model to historical data."""
                if len(data) < self.lookback_window:
                    return
                
                # Simple linear trend model
                x = np.arange(len(data))
                coeffs = np.polyfit(x, data, 1)
                self.model = coeffs
                
            def predict_next(self, data: np.ndarray, steps_ahead: int = 1) -> float:
                """Predict next value(s)."""
                if self.model is None:
                    return 0.0
                
                # Linear extrapolation
                next_x = len(data) + steps_ahead - 1
                prediction = self.model[0] * next_x + self.model[1]
                return prediction
        
        # Anomaly detector for metric patterns
        class MetricAnomalyDetector:
            def __init__(self):
                self.baseline_stats = {}
                self.anomaly_threshold = 3.0  # 3-sigma rule
                
            def fit(self, metrics: Dict[str, List[float]]):
                """Fit the detector to baseline metrics."""
                for metric_name, values in metrics.items():
                    if len(values) >= 10:
                        self.baseline_stats[metric_name] = {
                            'mean': np.mean(values),
                            'std': np.std(values),
                            'min': np.min(values),
                            'max': np.max(values)
                        }
            
            def detect_anomaly(self, metric_name: str, value: float) -> Tuple[bool, float]:
                """Detect if a metric value is anomalous."""
                if metric_name not in self.baseline_stats:
                    return False, 0.0
                
                stats = self.baseline_stats[metric_name]
                z_score = abs(value - stats['mean']) / (stats['std'] + 1e-8)
                
                is_anomaly = z_score > self.anomaly_threshold
                confidence = min(z_score / self.anomaly_threshold, 1.0)
                
                return is_anomaly, confidence
        
        # Failure pattern recognizer
        class FailurePatternRecognizer:
            def __init__(self):
                self.known_patterns = []
                self.pattern_threshold = 0.8
            
            def learn_pattern(self, metrics_before_failure: Dict[str, List[float]]):
                """Learn a failure pattern from historical data."""
                if any(len(values) >= 5 for values in metrics_before_failure.values()):
                    pattern = {}
                    for metric, values in metrics_before_failure.items():
                        if len(values) >= 5:
                            pattern[metric] = {
                                'trend': self._calculate_trend(values),
                                'volatility': np.std(values[-5:]),
                                'final_values': values[-3:]  # Last 3 values
                            }
                    
                    if pattern:
                        self.known_patterns.append(pattern)
            
            def recognize_pattern(self, current_metrics: Dict[str, List[float]]) -> float:
                """Recognize if current metrics match a known failure pattern."""
                if not self.known_patterns or not current_metrics:
                    return 0.0
                
                max_similarity = 0.0
                
                for pattern in self.known_patterns:
                    similarity = self._calculate_pattern_similarity(pattern, current_metrics)
                    max_similarity = max(max_similarity, similarity)
                
                return max_similarity
            
            def _calculate_trend(self, values: List[float]) -> str:
                """Calculate trend direction."""
                if len(values) < 2:
                    return "stable"
                
                recent = values[-3:] if len(values) >= 3 else values[-2:]
                if len(recent) < 2:
                    return "stable"
                
                slope = (recent[-1] - recent[0]) / len(recent)
                if slope > 0.1:
                    return "increasing"
                elif slope < -0.1:
                    return "decreasing"
                else:
                    return "stable"
            
            def _calculate_pattern_similarity(self, pattern: Dict, current_metrics: Dict) -> float:
                """Calculate similarity between pattern and current metrics."""
                similarities = []
                
                for metric_name, pattern_data in pattern.items():
                    if metric_name in current_metrics and len(current_metrics[metric_name]) >= 3:
                        current_values = current_metrics[metric_name]
                        
                        # Compare trends
                        current_trend = self._calculate_trend(current_values)
                        trend_match = 1.0 if current_trend == pattern_data['trend'] else 0.0
                        
                        # Compare volatility
                        current_volatility = np.std(current_values[-5:])
                        volatility_similarity = 1.0 - abs(current_volatility - pattern_data['volatility']) / max(current_volatility, pattern_data['volatility'], 1.0)
                        volatility_similarity = max(0.0, volatility_similarity)
                        
                        # Overall similarity for this metric
                        metric_similarity = (trend_match + volatility_similarity) / 2
                        similarities.append(metric_similarity)
                
                return np.mean(similarities) if similarities else 0.0
        
        self.failure_prediction_models.update({
            'time_series': TimeSeriesPredictor(),
            'anomaly_detector': MetricAnomalyDetector(),
            'pattern_recognizer': FailurePatternRecognizer()
        })
    
    def add_metric_data(self, metric_name: str, value: float, timestamp: Optional[datetime] = None) -> None:
        """Add metric data point for analysis."""
        timestamp = timestamp or datetime.now()
        
        self.metric_history[metric_name].append({
            'value': value,
            'timestamp': timestamp
        })
        
        # Update time series model periodically
        if len(self.metric_history[metric_name]) >= 20:
            values = [point['value'] for point in self.metric_history[metric_name]]
            self.failure_prediction_models['time_series'].fit(np.array(values))
    
    def record_failure(self, failure_type: str, metrics_before_failure: Dict[str, List[float]]) -> None:
        """Record a failure for pattern learning."""
        failure_record = {
            'timestamp': datetime.now().isoformat(),
            'failure_type': failure_type,
            'metrics_before_failure': metrics_before_failure
        }
        
        self.failure_history.append(failure_record)
        
        # Learn failure pattern
        pattern_recognizer = self.failure_prediction_models['pattern_recognizer']
        pattern_recognizer.learn_pattern(metrics_before_failure)
    
    async def predict_failure(self, current_metrics: Dict[str, HealthMetric]) -> Tuple[FailurePrediction, Dict[str, Any]]:
        """
        Predict potential system failures based on current metrics.
        
        Args:
            current_metrics: Current system health metrics
            
        Returns:
            Tuple of (prediction_level, prediction_details)
        """
        prediction_details = {
            'timestamp': datetime.now().isoformat(),
            'analysis': {},
            'recommendations': [],
            'confidence_scores': {}
        }
        
        failure_indicators = []
        
        # Time series analysis
        for metric_name, metric in current_metrics.items():
            if metric_name in self.metric_history and len(self.metric_history[metric_name]) >= 10:
                values = [point['value'] for point in self.metric_history[metric_name]]
                
                # Predict next value
                predictor = self.failure_prediction_models['time_series']
                predictor.fit(np.array(values))
                predicted_value = predictor.predict_next(np.array(values), steps_ahead=5)
                
                # Check if prediction exceeds thresholds
                if predicted_value >= metric.threshold_critical:
                    failure_indicators.append(0.9)
                    prediction_details['analysis'][metric_name] = {
                        'type': 'time_series',
                        'current': metric.value,
                        'predicted': predicted_value,
                        'threshold_critical': metric.threshold_critical,
                        'concern': 'Predicted to exceed critical threshold'
                    }
                elif predicted_value >= metric.threshold_warning:
                    failure_indicators.append(0.6)
                    prediction_details['analysis'][metric_name] = {
                        'type': 'time_series',
                        'current': metric.value,
                        'predicted': predicted_value,
                        'threshold_warning': metric.threshold_warning,
                        'concern': 'Predicted to exceed warning threshold'
                    }
        
        # Anomaly detection
        anomaly_detector = self.failure_prediction_models['anomaly_detector']
        
        # Fit detector if we have enough data
        if len(self.metric_history) > 0:
            baseline_data = {}
            for metric_name, history in self.metric_history.items():
                if len(history) >= 20:
                    baseline_data[metric_name] = [point['value'] for point in list(history)[-50:]]
            
            if baseline_data:
                anomaly_detector.fit(baseline_data)
        
        # Check current metrics for anomalies
        for metric_name, metric in current_metrics.items():
            is_anomaly, confidence = anomaly_detector.detect_anomaly(metric_name, metric.value)
            
            if is_anomaly:
                failure_indicators.append(confidence * 0.8)  # Weight anomaly detection
                prediction_details['analysis'][metric_name] = prediction_details['analysis'].get(metric_name, {})
                prediction_details['analysis'][metric_name].update({
                    'anomaly_detected': True,
                    'anomaly_confidence': confidence,
                    'concern': f'Anomalous value detected (confidence: {confidence:.2f})'
                })
        
        # Pattern recognition
        pattern_recognizer = self.failure_prediction_models['pattern_recognizer']
        current_metric_values = {}
        for metric_name, metric in current_metrics.items():
            if metric_name in self.metric_history:
                values = [point['value'] for point in list(self.metric_history[metric_name])[-10:]]
                if values:
                    current_metric_values[metric_name] = values
        
        pattern_similarity = pattern_recognizer.recognize_pattern(current_metric_values)
        if pattern_similarity > 0.5:
            failure_indicators.append(pattern_similarity)
            prediction_details['analysis']['pattern_recognition'] = {
                'similarity_score': pattern_similarity,
                'concern': f'Current metrics match known failure pattern (similarity: {pattern_similarity:.2f})'
            }
        
        # Calculate overall failure probability
        if failure_indicators:
            failure_probability = np.mean(failure_indicators)
        else:
            failure_probability = 0.0
        
        prediction_details['confidence_scores']['overall_failure_probability'] = failure_probability
        
        # Determine prediction level
        if failure_probability >= self.prediction_thresholds[FailurePrediction.IMMINENT]:
            prediction_level = FailurePrediction.IMMINENT
            prediction_details['recommendations'].extend([
                "Immediate intervention required",
                "Consider emergency shutdown if critical systems affected",
                "Activate disaster recovery procedures"
            ])
        elif failure_probability >= self.prediction_thresholds[FailurePrediction.HIGH]:
            prediction_level = FailurePrediction.HIGH
            prediction_details['recommendations'].extend([
                "Schedule maintenance window urgently",
                "Prepare rollback procedures",
                "Increase monitoring frequency"
            ])
        elif failure_probability >= self.prediction_thresholds[FailurePrediction.MEDIUM]:
            prediction_level = FailurePrediction.MEDIUM
            prediction_details['recommendations'].extend([
                "Investigate concerning metrics",
                "Consider preventive actions",
                "Monitor trends closely"
            ])
        elif failure_probability >= self.prediction_thresholds[FailurePrediction.LOW]:
            prediction_level = FailurePrediction.LOW
            prediction_details['recommendations'].append("Continue normal monitoring")
        else:
            prediction_level = FailurePrediction.LOW
            prediction_details['recommendations'].append("No immediate concerns detected")
        
        return prediction_level, prediction_details
    
    def get_prediction_summary(self) -> Dict[str, Any]:
        """Get prediction system summary."""
        return {
            "tracked_metrics": len(self.metric_history),
            "total_data_points": sum(len(history) for history in self.metric_history.values()),
            "recorded_failures": len(self.failure_history),
            "known_failure_patterns": len(self.failure_prediction_models['pattern_recognizer'].known_patterns),
            "prediction_thresholds": {k.value: v for k, v in self.prediction_thresholds.items()}
        }


class AutoRecoveryOrchestrator:
    """
    Automated recovery orchestration system.
    
    Features:
    - Dynamic recovery plan selection
    - Rollback mechanisms with state preservation
    - Cascade failure prevention
    - Recovery success validation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize auto-recovery orchestrator."""
        self.config = config or {}
        
        # Recovery plans repository
        self.recovery_plans: Dict[str, RecoveryPlan] = {}
        self.active_recoveries: Dict[str, Dict[str, Any]] = {}
        
        # State management
        self.system_snapshots: Dict[str, Dict[str, Any]] = {}
        self.recovery_history: List[Dict[str, Any]] = []
        
        # Recovery executors
        self.recovery_executors: Dict[RecoveryAction, Callable] = {}
        
        # Initialize default recovery plans and executors
        self._initialize_recovery_plans()
        self._initialize_recovery_executors()
    
    def _initialize_recovery_plans(self):
        """Initialize default recovery plans."""
        # High CPU usage recovery plan
        cpu_recovery_plan = RecoveryPlan(
            plan_id="cpu_high_usage",
            trigger_condition="cpu_usage > 80%",
            recovery_actions=[
                {"action": RecoveryAction.REDUCE_LOAD, "priority": 1, "timeout": 30},
                {"action": RecoveryAction.SCALE_RESOURCES, "priority": 2, "timeout": 60},
                {"action": RecoveryAction.RESTART_COMPONENT, "priority": 3, "timeout": 120}
            ],
            estimated_recovery_time=60.0,
            success_criteria={"cpu_usage": {"max": 70}},
            rollback_plan=[
                {"action": "restore_previous_configuration", "timeout": 30}
            ],
            priority=2
        )
        
        # Memory exhaustion recovery plan
        memory_recovery_plan = RecoveryPlan(
            plan_id="memory_exhaustion",
            trigger_condition="memory_usage > 90%",
            recovery_actions=[
                {"action": RecoveryAction.RESTART_COMPONENT, "priority": 1, "timeout": 60},
                {"action": RecoveryAction.SCALE_RESOURCES, "priority": 2, "timeout": 120},
                {"action": RecoveryAction.SWITCH_TO_BACKUP, "priority": 3, "timeout": 30}
            ],
            estimated_recovery_time=90.0,
            success_criteria={"memory_usage": {"max": 80}},
            rollback_plan=[
                {"action": "restore_backup_state", "timeout": 60}
            ],
            priority=1
        )
        
        # Model performance degradation recovery plan
        model_recovery_plan = RecoveryPlan(
            plan_id="model_performance_degradation",
            trigger_condition="model_accuracy < 80% OR prediction_time > 5s",
            recovery_actions=[
                {"action": RecoveryAction.ROLLBACK_MODEL, "priority": 1, "timeout": 30},
                {"action": RecoveryAction.RETRAIN_MODEL, "priority": 2, "timeout": 1800},  # 30 minutes
                {"action": RecoveryAction.SWITCH_TO_BACKUP, "priority": 3, "timeout": 15}
            ],
            estimated_recovery_time=120.0,
            success_criteria={"model_accuracy": {"min": 80}, "prediction_time": {"max": 3}},
            rollback_plan=[
                {"action": "restore_previous_model", "timeout": 30}
            ],
            priority=1
        )
        
        # Cascade failure prevention plan
        cascade_prevention_plan = RecoveryPlan(
            plan_id="cascade_failure_prevention",
            trigger_condition="multiple_component_failures OR failure_rate > 50%",
            recovery_actions=[
                {"action": RecoveryAction.EMERGENCY_SHUTDOWN, "priority": 1, "timeout": 10},
                {"action": RecoveryAction.SWITCH_TO_BACKUP, "priority": 2, "timeout": 30},
                {"action": RecoveryAction.RESTART_COMPONENT, "priority": 3, "timeout": 180}
            ],
            estimated_recovery_time=300.0,
            success_criteria={"system_status": "stable", "failure_rate": {"max": 10}},
            priority=1
        )
        
        self.recovery_plans.update({
            plan.plan_id: plan for plan in [
                cpu_recovery_plan,
                memory_recovery_plan,
                model_recovery_plan,
                cascade_prevention_plan
            ]
        })
    
    def _initialize_recovery_executors(self):
        """Initialize recovery action executors."""
        self.recovery_executors = {
            RecoveryAction.RESTART_COMPONENT: self._execute_restart_component,
            RecoveryAction.ROLLBACK_MODEL: self._execute_rollback_model,
            RecoveryAction.SCALE_RESOURCES: self._execute_scale_resources,
            RecoveryAction.SWITCH_TO_BACKUP: self._execute_switch_to_backup,
            RecoveryAction.RETRAIN_MODEL: self._execute_retrain_model,
            RecoveryAction.REDUCE_LOAD: self._execute_reduce_load,
            RecoveryAction.EMERGENCY_SHUTDOWN: self._execute_emergency_shutdown
        }
    
    async def execute_recovery(self, plan_id: str, context: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        """
        Execute a recovery plan.
        
        Args:
            plan_id: ID of the recovery plan to execute
            context: Current system context and metrics
            
        Returns:
            Tuple of (success, recovery_details)
        """
        plan = self.recovery_plans.get(plan_id)
        if not plan:
            return False, {"error": f"Recovery plan '{plan_id}' not found"}
        
        recovery_id = f"{plan_id}_{int(time.time())}"
        
        recovery_details = {
            "recovery_id": recovery_id,
            "plan_id": plan_id,
            "start_time": datetime.now().isoformat(),
            "actions_executed": [],
            "success": False,
            "error_messages": []
        }
        
        # Store initial system state for potential rollback
        initial_state = await self._capture_system_snapshot()
        self.system_snapshots[recovery_id] = initial_state
        
        # Mark recovery as active
        self.active_recoveries[recovery_id] = {
            "plan": plan,
            "context": context,
            "start_time": time.time(),
            "status": "executing"
        }
        
        try:
            logger.info(f"Starting recovery execution: {recovery_id}")
            
            # Sort actions by priority
            sorted_actions = sorted(plan.recovery_actions, key=lambda x: x["priority"])
            
            # Execute recovery actions
            for action_config in sorted_actions:
                action_type = action_config["action"]
                timeout = action_config.get("timeout", 60)
                
                try:
                    # Execute action with timeout
                    action_result = await asyncio.wait_for(
                        self._execute_recovery_action(action_type, context, recovery_id),
                        timeout=timeout
                    )
                    
                    recovery_details["actions_executed"].append({
                        "action": action_type.value,
                        "success": action_result.get("success", False),
                        "details": action_result.get("details", ""),
                        "timestamp": datetime.now().isoformat()
                    })
                    
                    if action_result.get("success", False):
                        logger.info(f"Recovery action {action_type.value} succeeded")
                        
                        # Check if this action resolved the issue
                        if await self._check_recovery_success(plan.success_criteria, context):
                            logger.info(f"Recovery successful after action: {action_type.value}")
                            recovery_details["success"] = True
                            break
                    else:
                        logger.warning(f"Recovery action {action_type.value} failed: {action_result.get('error', 'Unknown error')}")
                        recovery_details["error_messages"].append(f"{action_type.value}: {action_result.get('error', 'Unknown error')}")
                
                except asyncio.TimeoutError:
                    error_msg = f"Recovery action {action_type.value} timed out after {timeout}s"
                    logger.error(error_msg)
                    recovery_details["error_messages"].append(error_msg)
                    recovery_details["actions_executed"].append({
                        "action": action_type.value,
                        "success": False,
                        "details": "Timed out",
                        "timestamp": datetime.now().isoformat()
                    })
                
                except Exception as e:
                    error_msg = f"Recovery action {action_type.value} failed: {str(e)}"
                    logger.error(error_msg)
                    recovery_details["error_messages"].append(error_msg)
                    recovery_details["actions_executed"].append({
                        "action": action_type.value,
                        "success": False,
                        "details": str(e),
                        "timestamp": datetime.now().isoformat()
                    })
            
            # Final success check
            if not recovery_details["success"]:
                recovery_details["success"] = await self._check_recovery_success(plan.success_criteria, context)
            
            # If recovery failed and rollback is available, attempt rollback
            if not recovery_details["success"] and plan.rollback_plan:
                logger.info("Recovery failed, attempting rollback")
                rollback_result = await self._execute_rollback(plan.rollback_plan, recovery_id)
                recovery_details["rollback_executed"] = rollback_result
            
            recovery_details["end_time"] = datetime.now().isoformat()
            recovery_details["duration_seconds"] = time.time() - self.active_recoveries[recovery_id]["start_time"]
            
            # Update active recoveries status
            self.active_recoveries[recovery_id]["status"] = "completed" if recovery_details["success"] else "failed"
            
            # Record in history
            self.recovery_history.append(recovery_details)
            
            logger.info(f"Recovery execution completed: {recovery_id}, Success: {recovery_details['success']}")
            
            return recovery_details["success"], recovery_details
        
        except Exception as e:
            logger.error(f"Recovery execution failed with exception: {e}")
            recovery_details["error_messages"].append(f"Execution failed: {str(e)}")
            recovery_details["success"] = False
            recovery_details["end_time"] = datetime.now().isoformat()
            
            # Update status
            if recovery_id in self.active_recoveries:
                self.active_recoveries[recovery_id]["status"] = "failed"
            
            return False, recovery_details
        
        finally:
            # Clean up
            if recovery_id in self.active_recoveries:
                del self.active_recoveries[recovery_id]
    
    async def _execute_recovery_action(self, action_type: RecoveryAction, 
                                     context: Dict[str, Any], recovery_id: str) -> Dict[str, Any]:
        """Execute a specific recovery action."""
        executor = self.recovery_executors.get(action_type)
        if not executor:
            return {"success": False, "error": f"No executor found for action: {action_type.value}"}
        
        try:
            return await executor(context, recovery_id)
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    async def _execute_restart_component(self, context: Dict[str, Any], recovery_id: str) -> Dict[str, Any]:
        """Execute component restart."""
        component = context.get("component", "inference_engine")
        
        try:
            logger.info(f"Restarting component: {component}")
            
            # Simulate component restart
            await asyncio.sleep(2)
            
            # Clear memory/cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                "success": True,
                "details": f"Component {component} restarted successfully"
            }
        except Exception as e:
            return {"success": False, "error": f"Failed to restart {component}: {str(e)}"}
    
    async def _execute_rollback_model(self, context: Dict[str, Any], recovery_id: str) -> Dict[str, Any]:
        """Execute model rollback."""
        try:
            logger.info("Rolling back to previous model version")
            
            # Simulate model rollback
            await asyncio.sleep(1)
            
            # In real implementation, this would restore a previous model checkpoint
            model_version = context.get("previous_model_version", "v1.0")
            
            return {
                "success": True,
                "details": f"Rolled back to model version: {model_version}"
            }
        except Exception as e:
            return {"success": False, "error": f"Model rollback failed: {str(e)}"}
    
    async def _execute_scale_resources(self, context: Dict[str, Any], recovery_id: str) -> Dict[str, Any]:
        """Execute resource scaling."""
        try:
            logger.info("Scaling system resources")
            
            # Simulate resource scaling
            await asyncio.sleep(3)
            
            # In real implementation, this would trigger auto-scaling
            scale_factor = context.get("scale_factor", 1.5)
            
            return {
                "success": True,
                "details": f"Resources scaled by factor: {scale_factor}"
            }
        except Exception as e:
            return {"success": False, "error": f"Resource scaling failed: {str(e)}"}
    
    async def _execute_switch_to_backup(self, context: Dict[str, Any], recovery_id: str) -> Dict[str, Any]:
        """Execute switch to backup system."""
        try:
            logger.info("Switching to backup system")
            
            # Simulate backup switch
            await asyncio.sleep(1)
            
            backup_system = context.get("backup_system", "backup_instance")
            
            return {
                "success": True,
                "details": f"Switched to backup system: {backup_system}"
            }
        except Exception as e:
            return {"success": False, "error": f"Backup switch failed: {str(e)}"}
    
    async def _execute_retrain_model(self, context: Dict[str, Any], recovery_id: str) -> Dict[str, Any]:
        """Execute model retraining."""
        try:
            logger.info("Starting model retraining")
            
            # Simulate model retraining (this would be a long process)
            await asyncio.sleep(5)  # Simulate quick retraining for demo
            
            return {
                "success": True,
                "details": "Model retraining completed successfully"
            }
        except Exception as e:
            return {"success": False, "error": f"Model retraining failed: {str(e)}"}
    
    async def _execute_reduce_load(self, context: Dict[str, Any], recovery_id: str) -> Dict[str, Any]:
        """Execute load reduction."""
        try:
            logger.info("Reducing system load")
            
            # Simulate load reduction
            await asyncio.sleep(1)
            
            reduction_percentage = context.get("load_reduction", 50)
            
            return {
                "success": True,
                "details": f"System load reduced by {reduction_percentage}%"
            }
        except Exception as e:
            return {"success": False, "error": f"Load reduction failed: {str(e)}"}
    
    async def _execute_emergency_shutdown(self, context: Dict[str, Any], recovery_id: str) -> Dict[str, Any]:
        """Execute emergency shutdown."""
        try:
            logger.critical("Executing emergency shutdown")
            
            # Simulate emergency shutdown
            await asyncio.sleep(1)
            
            return {
                "success": True,
                "details": "Emergency shutdown executed successfully"
            }
        except Exception as e:
            return {"success": False, "error": f"Emergency shutdown failed: {str(e)}"}
    
    async def _capture_system_snapshot(self) -> Dict[str, Any]:
        """Capture current system state for rollback."""
        try:
            snapshot = {
                "timestamp": datetime.now().isoformat(),
                "system_metrics": {
                    "cpu_percent": psutil.cpu_percent(),
                    "memory_percent": psutil.virtual_memory().percent,
                    "disk_usage": psutil.disk_usage('/').percent if os.path.exists('/') else 0
                },
                "process_info": {
                    "pid": os.getpid(),
                    "memory_info": psutil.Process().memory_info()._asdict()
                }
            }
            
            # In real implementation, this would capture model states, configurations, etc.
            return snapshot
        
        except Exception as e:
            logger.error(f"Failed to capture system snapshot: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}
    
    async def _execute_rollback(self, rollback_plan: List[Dict[str, Any]], recovery_id: str) -> Dict[str, Any]:
        """Execute rollback plan."""
        rollback_result = {
            "success": False,
            "actions": [],
            "error_messages": []
        }
        
        try:
            for action in rollback_plan:
                action_name = action.get("action", "unknown")
                timeout = action.get("timeout", 30)
                
                try:
                    # Simulate rollback action
                    await asyncio.sleep(1)
                    
                    rollback_result["actions"].append({
                        "action": action_name,
                        "success": True,
                        "details": f"Rollback action {action_name} completed"
                    })
                
                except asyncio.TimeoutError:
                    error_msg = f"Rollback action {action_name} timed out"
                    rollback_result["error_messages"].append(error_msg)
                    rollback_result["actions"].append({
                        "action": action_name,
                        "success": False,
                        "details": "Timed out"
                    })
            
            rollback_result["success"] = len(rollback_result["error_messages"]) == 0
            
        except Exception as e:
            rollback_result["error_messages"].append(f"Rollback execution failed: {str(e)}")
        
        return rollback_result
    
    async def _check_recovery_success(self, success_criteria: Dict[str, Any], 
                                    context: Dict[str, Any]) -> bool:
        """Check if recovery was successful based on criteria."""
        try:
            # Get current system metrics (simplified)
            current_metrics = {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "model_accuracy": context.get("model_accuracy", 85),  # Mock value
                "prediction_time": context.get("prediction_time", 2),  # Mock value
                "system_status": context.get("system_status", "stable"),
                "failure_rate": context.get("failure_rate", 5)  # Mock value
            }
            
            # Check each criterion
            for metric_name, criteria in success_criteria.items():
                if metric_name not in current_metrics:
                    continue
                
                current_value = current_metrics[metric_name]
                
                # Check minimum thresholds
                if "min" in criteria and current_value < criteria["min"]:
                    return False
                
                # Check maximum thresholds
                if "max" in criteria and current_value > criteria["max"]:
                    return False
                
                # Check exact values
                if "equals" in criteria and current_value != criteria["equals"]:
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking recovery success: {e}")
            return False
    
    def add_recovery_plan(self, plan: RecoveryPlan) -> None:
        """Add a new recovery plan."""
        self.recovery_plans[plan.plan_id] = plan
        logger.info(f"Added recovery plan: {plan.plan_id}")
    
    def get_recovery_summary(self) -> Dict[str, Any]:
        """Get recovery system summary."""
        return {
            "total_recovery_plans": len(self.recovery_plans),
            "active_recoveries": len(self.active_recoveries),
            "recovery_history_count": len(self.recovery_history),
            "available_recovery_actions": [action.value for action in RecoveryAction],
            "recent_recoveries": [
                {
                    "recovery_id": entry["recovery_id"],
                    "plan_id": entry["plan_id"],
                    "success": entry["success"],
                    "start_time": entry["start_time"]
                }
                for entry in self.recovery_history[-10:]
            ]
        }


class SelfHealingSystem:
    """
    Comprehensive self-healing system orchestrating all components.
    
    Features:
    - Continuous health monitoring
    - Predictive failure detection
    - Automated recovery orchestration
    - Learning from recovery experiences
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize self-healing system."""
        self.config = config or {}
        
        # Initialize components
        self.failure_detector = PredictiveFailureDetector(config.get("failure_detection", {}))
        self.recovery_orchestrator = AutoRecoveryOrchestrator(config.get("recovery", {}))
        
        # System state
        self.system_health: Dict[str, HealthMetric] = {}
        self.is_running = False
        self.background_tasks: Set[asyncio.Task] = set()
        
        # Monitoring configuration
        self.monitoring_interval = self.config.get("monitoring_interval", 10)  # seconds
        self.prediction_interval = self.config.get("prediction_interval", 30)  # seconds
        
        logger.info("Self-healing system initialized")
    
    async def start_self_healing(self) -> None:
        """Start the self-healing system."""
        if self.is_running:
            logger.warning("Self-healing system is already running")
            return
        
        self.is_running = True
        logger.info("Starting self-healing system")
        
        # Start background tasks
        tasks = [
            self._health_monitoring_task(),
            self._failure_prediction_task(),
            self._recovery_orchestration_task(),
            self._learning_task()
        ]
        
        for task_coro in tasks:
            task = asyncio.create_task(task_coro)
            self.background_tasks.add(task)
            task.add_done_callback(self.background_tasks.discard)
        
        logger.info("Self-healing system started")
    
    async def stop_self_healing(self) -> None:
        """Stop the self-healing system."""
        if not self.is_running:
            logger.warning("Self-healing system is not running")
            return
        
        self.is_running = False
        logger.info("Stopping self-healing system")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        logger.info("Self-healing system stopped")
    
    async def _health_monitoring_task(self) -> None:
        """Background task for continuous health monitoring."""
        while self.is_running:
            try:
                await self._collect_system_metrics()
                await asyncio.sleep(self.monitoring_interval)
            except Exception as e:
                logger.error(f"Error in health monitoring task: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def _failure_prediction_task(self) -> None:
        """Background task for failure prediction."""
        while self.is_running:
            try:
                if self.system_health:
                    prediction_level, prediction_details = await self.failure_detector.predict_failure(self.system_health)
                    
                    if prediction_level in [FailurePrediction.HIGH, FailurePrediction.IMMINENT]:
                        logger.warning(f"Failure prediction: {prediction_level.value}")
                        logger.info(f"Prediction details: {json.dumps(prediction_details, indent=2)}")
                        
                        # Trigger proactive recovery if prediction is imminent
                        if prediction_level == FailurePrediction.IMMINENT:
                            await self._trigger_proactive_recovery(prediction_details)
                
                await asyncio.sleep(self.prediction_interval)
            except Exception as e:
                logger.error(f"Error in failure prediction task: {e}")
                await asyncio.sleep(self.prediction_interval)
    
    async def _recovery_orchestration_task(self) -> None:
        """Background task for recovery orchestration."""
        while self.is_running:
            try:
                # Check if any metrics are in critical state
                critical_metrics = [
                    name for name, metric in self.system_health.items()
                    if metric.get_status() == HealthStatus.CRITICAL
                ]
                
                if critical_metrics:
                    logger.critical(f"Critical metrics detected: {critical_metrics}")
                    await self._handle_critical_metrics(critical_metrics)
                
                await asyncio.sleep(5)  # More frequent checks for critical issues
            except Exception as e:
                logger.error(f"Error in recovery orchestration task: {e}")
                await asyncio.sleep(5)
    
    async def _learning_task(self) -> None:
        """Background task for learning from recovery experiences."""
        while self.is_running:
            try:
                # Analyze recent recovery history for patterns
                recent_recoveries = self.recovery_orchestrator.recovery_history[-10:]
                
                if recent_recoveries:
                    await self._analyze_recovery_patterns(recent_recoveries)
                
                await asyncio.sleep(300)  # Every 5 minutes
            except Exception as e:
                logger.error(f"Error in learning task: {e}")
                await asyncio.sleep(300)
    
    async def _collect_system_metrics(self) -> None:
        """Collect current system metrics."""
        try:
            # System resource metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            
            # Disk usage if available
            disk_percent = 0
            try:
                disk_percent = psutil.disk_usage('/').percent
            except:
                pass
            
            # Update health metrics
            self.system_health.update({
                "cpu_usage": HealthMetric(
                    name="cpu_usage",
                    value=cpu_percent,
                    threshold_warning=70.0,
                    threshold_critical=85.0
                ),
                "memory_usage": HealthMetric(
                    name="memory_usage",
                    value=memory_percent,
                    threshold_warning=80.0,
                    threshold_critical=90.0
                ),
                "disk_usage": HealthMetric(
                    name="disk_usage",
                    value=disk_percent,
                    threshold_warning=80.0,
                    threshold_critical=95.0
                )
            })
            
            # Add mock model performance metrics
            # In real implementation, these would come from the actual model
            self.system_health.update({
                "model_accuracy": HealthMetric(
                    name="model_accuracy",
                    value=85.0,  # Mock value
                    threshold_warning=75.0,
                    threshold_critical=65.0
                ),
                "prediction_time": HealthMetric(
                    name="prediction_time",
                    value=2.5,  # Mock value in seconds
                    threshold_warning=3.0,
                    threshold_critical=5.0
                )
            })
            
            # Send metrics to failure detector
            for name, metric in self.system_health.items():
                self.failure_detector.add_metric_data(name, metric.value)
        
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    async def _trigger_proactive_recovery(self, prediction_details: Dict[str, Any]) -> None:
        """Trigger proactive recovery based on failure predictions."""
        logger.info("Triggering proactive recovery")
        
        # Determine appropriate recovery plan based on predicted failure
        analysis = prediction_details.get("analysis", {})
        
        # Check for specific failure patterns
        if any("cpu" in key.lower() for key in analysis.keys()):
            plan_id = "cpu_high_usage"
        elif any("memory" in key.lower() for key in analysis.keys()):
            plan_id = "memory_exhaustion"
        elif any("model" in key.lower() for key in analysis.keys()):
            plan_id = "model_performance_degradation"
        else:
            plan_id = "cpu_high_usage"  # Default plan
        
        context = {
            "prediction_triggered": True,
            "prediction_details": prediction_details,
            "current_metrics": {name: metric.value for name, metric in self.system_health.items()}
        }
        
        success, recovery_details = await self.recovery_orchestrator.execute_recovery(plan_id, context)
        
        if success:
            logger.info(f"Proactive recovery successful: {recovery_details['recovery_id']}")
        else:
            logger.error(f"Proactive recovery failed: {recovery_details.get('error_messages', [])}")
    
    async def _handle_critical_metrics(self, critical_metrics: List[str]) -> None:
        """Handle metrics that are in critical state."""
        logger.critical(f"Handling critical metrics: {critical_metrics}")
        
        # Determine recovery plan based on critical metrics
        if "memory_usage" in critical_metrics:
            plan_id = "memory_exhaustion"
        elif "cpu_usage" in critical_metrics:
            plan_id = "cpu_high_usage"
        elif "model_accuracy" in critical_metrics or "prediction_time" in critical_metrics:
            plan_id = "model_performance_degradation"
        elif len(critical_metrics) > 2:
            plan_id = "cascade_failure_prevention"
        else:
            plan_id = "cpu_high_usage"  # Default
        
        context = {
            "critical_metrics": critical_metrics,
            "current_metrics": {name: metric.value for name, metric in self.system_health.items()},
            "trigger_reason": "critical_threshold_exceeded"
        }
        
        success, recovery_details = await self.recovery_orchestrator.execute_recovery(plan_id, context)
        
        if success:
            logger.info(f"Critical recovery successful: {recovery_details['recovery_id']}")
        else:
            logger.error(f"Critical recovery failed: {recovery_details.get('error_messages', [])}")
    
    async def _analyze_recovery_patterns(self, recent_recoveries: List[Dict[str, Any]]) -> None:
        """Analyze recovery patterns for learning and optimization."""
        try:
            # Calculate success rates by plan
            plan_success_rates = defaultdict(lambda: {"success": 0, "total": 0})
            
            for recovery in recent_recoveries:
                plan_id = recovery["plan_id"]
                plan_success_rates[plan_id]["total"] += 1
                if recovery["success"]:
                    plan_success_rates[plan_id]["success"] += 1
            
            # Log insights
            for plan_id, stats in plan_success_rates.items():
                success_rate = stats["success"] / stats["total"] * 100
                logger.info(f"Recovery plan {plan_id}: {success_rate:.1f}% success rate ({stats['success']}/{stats['total']})")
                
                # If success rate is low, consider plan optimization
                if success_rate < 50 and stats["total"] >= 3:
                    logger.warning(f"Low success rate for plan {plan_id}, consider optimization")
            
        except Exception as e:
            logger.error(f"Error analyzing recovery patterns: {e}")
    
    def get_system_health_status(self) -> Dict[str, Any]:
        """Get comprehensive system health status."""
        overall_status = HealthStatus.HEALTHY
        critical_metrics = []
        warning_metrics = []
        
        for name, metric in self.system_health.items():
            status = metric.get_status()
            if status == HealthStatus.CRITICAL:
                overall_status = HealthStatus.CRITICAL
                critical_metrics.append(name)
            elif status == HealthStatus.WARNING and overall_status not in [HealthStatus.CRITICAL]:
                overall_status = HealthStatus.WARNING
                warning_metrics.append(name)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": overall_status.value,
            "is_self_healing_active": self.is_running,
            "metrics": {name: metric.to_dict() for name, metric in self.system_health.items()},
            "critical_metrics": critical_metrics,
            "warning_metrics": warning_metrics,
            "failure_detector_summary": self.failure_detector.get_prediction_summary(),
            "recovery_orchestrator_summary": self.recovery_orchestrator.get_recovery_summary(),
            "background_tasks_count": len(self.background_tasks)
        }


# Global self-healing system instance
_self_healing_system: Optional[SelfHealingSystem] = None


def get_self_healing_system(config: Optional[Dict[str, Any]] = None) -> SelfHealingSystem:
    """Get or create the global self-healing system."""
    global _self_healing_system
    
    if _self_healing_system is None:
        _self_healing_system = SelfHealingSystem(config)
    
    return _self_healing_system


# Example usage
async def start_self_healing_with_config():
    """Start self-healing system with production configuration."""
    config = {
        "monitoring_interval": 10,  # Monitor every 10 seconds
        "prediction_interval": 30,  # Predict failures every 30 seconds
        "failure_detection": {
            "prediction_thresholds": {
                "low": 0.2,
                "medium": 0.4,
                "high": 0.6,
                "imminent": 0.8
            }
        },
        "recovery": {
            "max_concurrent_recoveries": 3,
            "recovery_timeout": 300  # 5 minutes
        }
    }
    
    system = get_self_healing_system(config)
    await system.start_self_healing()
    
    logger.info("Self-healing system started with production configuration")
    return system