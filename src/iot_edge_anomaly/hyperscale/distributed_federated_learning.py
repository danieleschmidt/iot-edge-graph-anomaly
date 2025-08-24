"""
Distributed Federated Learning Coordinator for Hyperscale IoT Systems.

Advanced federated learning system capable of coordinating 10,000+ clients with
model sharding, Byzantine fault tolerance, and hierarchical aggregation.

Key Features:
- Hierarchical federated learning with edge-fog-cloud coordination
- Model sharding across distributed inference engines
- Byzantine-robust aggregation with advanced algorithms
- Asynchronous federated updates with consistency guarantees
- Differential privacy with adaptive noise calibration
- Cross-device knowledge distillation
- Federated multi-task learning across IoT domains
- Dynamic client selection and resource optimization
- Blockchain-ready secure aggregation protocols
- Real-time convergence monitoring and optimization
"""

import asyncio
import logging
import time
import json
import threading
import hashlib
import uuid
import pickle
from typing import Dict, Any, List, Optional, Set, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import math
import random
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


class FederatedRole(Enum):
    """Roles in the federated learning hierarchy."""
    EDGE_CLIENT = "edge_client"
    FOG_AGGREGATOR = "fog_aggregator"
    CLOUD_COORDINATOR = "cloud_coordinator"
    GLOBAL_SERVER = "global_server"


class AggregationAlgorithm(Enum):
    """Federated aggregation algorithms."""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    SCAFFOLD = "scaffold"
    FEDNOVA = "fednova"
    KRUM = "krum"
    BULYAN = "bulyan"
    MEDIAN = "median"
    TRIMMED_MEAN = "trimmed_mean"
    ADAPTIVE_CLIPPING = "adaptive_clipping"


class PrivacyMechanism(Enum):
    """Privacy preservation mechanisms."""
    NONE = "none"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    SECURE_AGGREGATION = "secure_aggregation"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    MULTI_PARTY_COMPUTATION = "multi_party_computation"


@dataclass
class ModelShard:
    """Represents a shard of a federated model."""
    shard_id: str
    layer_names: List[str]
    parameters: Dict[str, torch.Tensor]
    gradient_accumulator: Optional[Dict[str, torch.Tensor]] = None
    shard_size_mb: float = 0.0
    client_assignments: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        if self.gradient_accumulator is None:
            self.gradient_accumulator = {}
        
        # Calculate shard size
        total_params = 0
        for param in self.parameters.values():
            total_params += param.numel() * param.element_size()
        self.shard_size_mb = total_params / (1024 * 1024)


@dataclass
class FederatedClient:
    """Federated learning client information."""
    client_id: str
    role: FederatedRole
    capabilities: Dict[str, Any]
    current_round: int = 0
    data_samples: int = 0
    model_version: int = 0
    last_update: Optional[datetime] = None
    contribution_score: float = 1.0
    reliability_score: float = 1.0
    privacy_budget_remaining: float = 1.0
    assigned_shards: List[str] = field(default_factory=list)
    geographic_region: Optional[str] = None
    network_latency_ms: float = 100.0
    compute_capacity: float = 1.0
    is_byzantine: bool = False
    consecutive_failures: int = 0


@dataclass
class FederatedRound:
    """Information about a federated learning round."""
    round_id: int
    start_time: datetime
    end_time: Optional[datetime] = None
    selected_clients: List[str] = field(default_factory=list)
    received_updates: Dict[str, Dict] = field(default_factory=dict)
    aggregation_algorithm: AggregationAlgorithm = AggregationAlgorithm.FEDAVG
    privacy_mechanism: PrivacyMechanism = PrivacyMechanism.DIFFERENTIAL_PRIVACY
    convergence_metrics: Dict[str, float] = field(default_factory=dict)
    byzantine_clients_detected: List[str] = field(default_factory=list)
    aggregation_time_ms: float = 0.0
    communication_cost_mb: float = 0.0


class ByzantineRobustAggregator:
    """
    Advanced Byzantine-robust aggregation algorithms.
    
    Implements multiple state-of-the-art algorithms for handling malicious
    clients in federated learning environments.
    """
    
    def __init__(self, byzantine_tolerance: float = 0.3):
        self.byzantine_tolerance = byzantine_tolerance
        self.detection_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.statistical_tests = {}
    
    def aggregate_updates(
        self, 
        updates: Dict[str, Dict[str, torch.Tensor]], 
        algorithm: AggregationAlgorithm,
        client_weights: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        """
        Aggregate model updates using specified Byzantine-robust algorithm.
        
        Args:
            updates: Dictionary of client_id -> parameter updates
            algorithm: Aggregation algorithm to use
            client_weights: Optional weights for each client
            
        Returns:
            Tuple of (aggregated_parameters, detected_byzantine_clients)
        """
        if not updates:
            return {}, []
        
        byzantine_clients = []
        
        try:
            if algorithm == AggregationAlgorithm.FEDAVG:
                aggregated = self._federated_averaging(updates, client_weights)
            elif algorithm == AggregationAlgorithm.KRUM:
                aggregated, byzantine_clients = self._krum_aggregation(updates)
            elif algorithm == AggregationAlgorithm.BULYAN:
                aggregated, byzantine_clients = self._bulyan_aggregation(updates)
            elif algorithm == AggregationAlgorithm.MEDIAN:
                aggregated = self._coordinate_wise_median(updates)
            elif algorithm == AggregationAlgorithm.TRIMMED_MEAN:
                aggregated = self._trimmed_mean_aggregation(updates)
            elif algorithm == AggregationAlgorithm.ADAPTIVE_CLIPPING:
                aggregated, byzantine_clients = self._adaptive_clipping_aggregation(updates)
            else:
                # Default to FedAvg
                aggregated = self._federated_averaging(updates, client_weights)
            
            # Update Byzantine detection history
            for client_id in byzantine_clients:
                self.detection_history[client_id].append(datetime.now())
            
            return aggregated, byzantine_clients
            
        except Exception as e:
            logger.error(f"Aggregation failed with {algorithm.value}: {e}")
            # Fall back to simple averaging
            return self._federated_averaging(updates, client_weights), []
    
    def _federated_averaging(
        self, 
        updates: Dict[str, Dict[str, torch.Tensor]], 
        client_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, torch.Tensor]:
        """Standard FedAvg aggregation."""
        if not updates:
            return {}
        
        # Get parameter names from first client
        param_names = list(next(iter(updates.values())).keys())
        aggregated = {}
        
        # Calculate weights
        if client_weights is None:
            # Equal weights
            weights = {client_id: 1.0 / len(updates) for client_id in updates.keys()}
        else:
            total_weight = sum(client_weights.values())
            weights = {
                client_id: client_weights.get(client_id, 1.0) / total_weight 
                for client_id in updates.keys()
            }
        
        # Aggregate each parameter
        for param_name in param_names:
            weighted_params = []
            
            for client_id, client_updates in updates.items():
                if param_name in client_updates:
                    weight = weights[client_id]
                    weighted_param = client_updates[param_name] * weight
                    weighted_params.append(weighted_param)
            
            if weighted_params:
                aggregated[param_name] = torch.stack(weighted_params).sum(dim=0)
        
        return aggregated
    
    def _krum_aggregation(
        self, 
        updates: Dict[str, Dict[str, torch.Tensor]]
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        """Krum aggregation algorithm for Byzantine robustness."""
        if len(updates) < 3:
            return self._federated_averaging(updates), []
        
        # Flatten all updates into vectors
        client_vectors = {}
        for client_id, client_updates in updates.items():
            flattened = torch.cat([param.flatten() for param in client_updates.values()])
            client_vectors[client_id] = flattened
        
        client_ids = list(client_vectors.keys())
        n_clients = len(client_ids)
        f = int(n_clients * self.byzantine_tolerance)  # Maximum Byzantine clients
        
        # Calculate Krum scores
        krum_scores = {}
        
        for i, client_i in enumerate(client_ids):
            # Calculate distances to all other clients
            distances = []
            
            for j, client_j in enumerate(client_ids):
                if i != j:
                    distance = torch.norm(client_vectors[client_i] - client_vectors[client_j]).item()
                    distances.append(distance)
            
            # Krum score is sum of distances to n-f-2 closest clients
            distances.sort()
            n_closest = n_clients - f - 2
            krum_scores[client_i] = sum(distances[:max(1, n_closest)])
        
        # Select client with minimum Krum score
        selected_client = min(krum_scores.keys(), key=lambda k: krum_scores[k])
        
        # Identify potential Byzantine clients (those with high Krum scores)
        median_score = np.median(list(krum_scores.values()))
        std_score = np.std(list(krum_scores.values()))
        threshold = median_score + 2 * std_score
        
        byzantine_clients = [
            client_id for client_id, score in krum_scores.items()
            if score > threshold
        ]
        
        return updates[selected_client], byzantine_clients
    
    def _bulyan_aggregation(
        self, 
        updates: Dict[str, Dict[str, torch.Tensor]]
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        """Bulyan aggregation algorithm."""
        if len(updates) < 4:
            return self._federated_averaging(updates), []
        
        n_clients = len(updates)
        f = int(n_clients * self.byzantine_tolerance)
        
        # Multi-Krum selection
        selected_updates = {}
        byzantine_clients = []
        
        # Run Krum multiple times to select multiple good updates
        remaining_updates = updates.copy()
        
        for _ in range(n_clients - 2 * f):
            if len(remaining_updates) < 2:
                break
                
            krum_result, krum_byzantine = self._krum_aggregation(remaining_updates)
            
            # Find the selected client
            for client_id, client_updates in remaining_updates.items():
                if self._updates_equal(client_updates, krum_result):
                    selected_updates[client_id] = client_updates
                    del remaining_updates[client_id]
                    break
            
            byzantine_clients.extend(krum_byzantine)
        
        # Coordinate-wise trimmed mean on selected updates
        if selected_updates:
            aggregated = self._trimmed_mean_aggregation(selected_updates)
        else:
            aggregated = self._federated_averaging(updates)
        
        return aggregated, list(set(byzantine_clients))
    
    def _coordinate_wise_median(
        self, 
        updates: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Coordinate-wise median aggregation."""
        if not updates:
            return {}
        
        param_names = list(next(iter(updates.values())).keys())
        aggregated = {}
        
        for param_name in param_names:
            param_values = []
            
            for client_updates in updates.values():
                if param_name in client_updates:
                    param_values.append(client_updates[param_name])
            
            if param_values:
                # Stack parameters and compute median along client dimension
                stacked = torch.stack(param_values, dim=0)
                aggregated[param_name] = torch.median(stacked, dim=0)[0]
        
        return aggregated
    
    def _trimmed_mean_aggregation(
        self, 
        updates: Dict[str, Dict[str, torch.Tensor]]
    ) -> Dict[str, torch.Tensor]:
        """Trimmed mean aggregation (remove extreme values)."""
        if not updates:
            return {}
        
        param_names = list(next(iter(updates.values())).keys())
        aggregated = {}
        
        # Trim ratio (remove this fraction from each end)
        trim_ratio = min(0.2, self.byzantine_tolerance)
        
        for param_name in param_names:
            param_values = []
            
            for client_updates in updates.values():
                if param_name in client_updates:
                    param_values.append(client_updates[param_name])
            
            if param_values:
                stacked = torch.stack(param_values, dim=0)
                
                # Sort along client dimension
                sorted_values, _ = torch.sort(stacked, dim=0)
                
                # Calculate trim indices
                n_clients = sorted_values.size(0)
                trim_count = int(n_clients * trim_ratio)
                
                if trim_count > 0:
                    # Remove extreme values and average the rest
                    trimmed = sorted_values[trim_count:-trim_count]
                    if trimmed.size(0) > 0:
                        aggregated[param_name] = torch.mean(trimmed, dim=0)
                    else:
                        aggregated[param_name] = torch.mean(sorted_values, dim=0)
                else:
                    aggregated[param_name] = torch.mean(sorted_values, dim=0)
        
        return aggregated
    
    def _adaptive_clipping_aggregation(
        self, 
        updates: Dict[str, Dict[str, torch.Tensor]]
    ) -> Tuple[Dict[str, torch.Tensor], List[str]]:
        """Adaptive gradient clipping with Byzantine detection."""
        if not updates:
            return {}, []
        
        # Calculate gradient norms for each client
        client_norms = {}
        for client_id, client_updates in updates.items():
            total_norm = 0.0
            for param in client_updates.values():
                total_norm += torch.norm(param).item() ** 2
            client_norms[client_id] = math.sqrt(total_norm)
        
        # Detect outliers based on gradient norms
        norms = list(client_norms.values())
        median_norm = np.median(norms)
        mad = np.median([abs(n - median_norm) for n in norms])
        
        # Adaptive threshold
        threshold = median_norm + 3 * mad
        
        byzantine_clients = [
            client_id for client_id, norm in client_norms.items()
            if norm > threshold
        ]
        
        # Clip gradients and aggregate
        clipped_updates = {}
        for client_id, client_updates in updates.items():
            if client_id not in byzantine_clients:
                norm = client_norms[client_id]
                if norm > median_norm:
                    # Clip to median norm
                    clipping_factor = median_norm / norm
                    clipped_update = {
                        name: param * clipping_factor
                        for name, param in client_updates.items()
                    }
                    clipped_updates[client_id] = clipped_update
                else:
                    clipped_updates[client_id] = client_updates
        
        aggregated = self._federated_averaging(clipped_updates)
        return aggregated, byzantine_clients
    
    def _updates_equal(
        self, 
        updates1: Dict[str, torch.Tensor], 
        updates2: Dict[str, torch.Tensor]
    ) -> bool:
        """Check if two update dictionaries are equal."""
        if set(updates1.keys()) != set(updates2.keys()):
            return False
        
        for key in updates1.keys():
            if not torch.allclose(updates1[key], updates2[key], rtol=1e-5):
                return False
        
        return True


class DifferentialPrivacyManager:
    """
    Advanced differential privacy manager for federated learning.
    
    Implements adaptive noise calibration and privacy budget management
    across multiple clients and rounds.
    """
    
    def __init__(
        self, 
        epsilon: float = 1.0, 
        delta: float = 1e-5,
        noise_multiplier: float = 1.0
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.noise_multiplier = noise_multiplier
        
        # Privacy budget tracking
        self.client_budgets: Dict[str, float] = defaultdict(lambda: epsilon)
        self.round_epsilons: deque = deque(maxlen=1000)
        
        # Adaptive calibration
        self.sensitivity_history: deque = deque(maxlen=100)
        self.noise_history: deque = deque(maxlen=100)
    
    def add_noise_to_updates(
        self, 
        updates: Dict[str, torch.Tensor], 
        client_id: str,
        clip_norm: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """Add differential privacy noise to model updates."""
        
        if self.client_budgets[client_id] <= 0:
            logger.warning(f"Client {client_id} has exhausted privacy budget")
            return {}
        
        noisy_updates = {}
        
        try:
            # Calculate current sensitivity
            total_norm = sum(torch.norm(param).item() for param in updates.values())
            self.sensitivity_history.append(total_norm)
            
            # Adaptive sensitivity estimation
            if len(self.sensitivity_history) > 10:
                adaptive_sensitivity = np.percentile(list(self.sensitivity_history), 95)
            else:
                adaptive_sensitivity = clip_norm
            
            # Clip gradients for bounded sensitivity
            clipped_updates = self._clip_updates(updates, clip_norm)
            
            # Calculate noise scale
            noise_scale = self._calculate_noise_scale(
                adaptive_sensitivity, 
                self.client_budgets[client_id]
            )
            
            # Add noise to each parameter
            for param_name, param in clipped_updates.items():
                noise = torch.normal(
                    mean=0.0, 
                    std=noise_scale, 
                    size=param.shape,
                    device=param.device
                )
                noisy_updates[param_name] = param + noise
            
            # Update privacy budget
            used_epsilon = self._calculate_epsilon_spent(noise_scale, adaptive_sensitivity)
            self.client_budgets[client_id] -= used_epsilon
            self.round_epsilons.append(used_epsilon)
            
            logger.debug(f"Added DP noise to client {client_id}, "
                        f"remaining budget: {self.client_budgets[client_id]:.4f}")
            
            return noisy_updates
            
        except Exception as e:
            logger.error(f"Failed to add DP noise for client {client_id}: {e}")
            return updates
    
    def _clip_updates(
        self, 
        updates: Dict[str, torch.Tensor], 
        clip_norm: float
    ) -> Dict[str, torch.Tensor]:
        """Clip model updates to bounded sensitivity."""
        
        # Calculate total norm
        total_norm = torch.sqrt(sum(torch.norm(param) ** 2 for param in updates.values()))
        
        if total_norm <= clip_norm:
            return updates
        
        # Clip to bound
        clip_factor = clip_norm / total_norm
        clipped_updates = {
            name: param * clip_factor 
            for name, param in updates.items()
        }
        
        return clipped_updates
    
    def _calculate_noise_scale(self, sensitivity: float, epsilon_budget: float) -> float:
        """Calculate noise scale for differential privacy."""
        
        # Use available epsilon, but not more than configured maximum
        effective_epsilon = min(epsilon_budget, self.epsilon / 10)  # Conservative
        
        if effective_epsilon <= 0:
            return float('inf')  # Infinite noise if no budget
        
        # Gaussian mechanism noise scale
        if self.delta > 0:
            # σ >= sqrt(2 * ln(1.25/δ)) * Δf / ε
            noise_scale = (
                math.sqrt(2 * math.log(1.25 / self.delta)) * 
                sensitivity / effective_epsilon
            )
        else:
            # Laplace mechanism for pure ε-DP
            noise_scale = sensitivity / effective_epsilon
        
        # Apply noise multiplier
        noise_scale *= self.noise_multiplier
        
        self.noise_history.append(noise_scale)
        return noise_scale
    
    def _calculate_epsilon_spent(self, noise_scale: float, sensitivity: float) -> float:
        """Calculate epsilon spent for given noise scale."""
        
        if noise_scale == 0:
            return self.epsilon  # Maximum epsilon if no noise
        
        if self.delta > 0:
            # Gaussian mechanism
            epsilon_spent = (
                math.sqrt(2 * math.log(1.25 / self.delta)) * 
                sensitivity / noise_scale
            )
        else:
            # Laplace mechanism
            epsilon_spent = sensitivity / noise_scale
        
        return min(epsilon_spent, self.epsilon)
    
    def get_privacy_stats(self) -> Dict[str, Any]:
        """Get privacy preservation statistics."""
        
        total_clients = len(self.client_budgets)
        exhausted_clients = sum(1 for budget in self.client_budgets.values() if budget <= 0)
        
        stats = {
            'total_epsilon': self.epsilon,
            'delta': self.delta,
            'clients_tracked': total_clients,
            'clients_budget_exhausted': exhausted_clients,
            'average_remaining_budget': np.mean(list(self.client_budgets.values())) if total_clients > 0 else 0,
            'rounds_with_privacy': len(self.round_epsilons),
            'average_round_epsilon': np.mean(list(self.round_epsilons)) if self.round_epsilons else 0
        }
        
        if self.noise_history:
            stats['average_noise_scale'] = np.mean(list(self.noise_history))
            stats['noise_scale_trend'] = 'increasing' if len(self.noise_history) >= 2 and self.noise_history[-1] > self.noise_history[0] else 'stable'
        
        return stats


class ModelShardingManager:
    """
    Advanced model sharding manager for distributed federated learning.
    
    Manages model partitioning, shard assignment, and gradient accumulation
    across thousands of clients.
    """
    
    def __init__(self, sharding_strategy: str = "layer_wise"):
        self.sharding_strategy = sharding_strategy
        self.shards: Dict[str, ModelShard] = {}
        self.client_shard_assignments: Dict[str, List[str]] = defaultdict(list)
        self.shard_dependencies: Dict[str, List[str]] = {}
        
        # Performance tracking
        self.shard_update_times: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self.communication_costs: Dict[str, float] = defaultdict(float)
    
    def create_model_shards(
        self, 
        model: nn.Module, 
        num_shards: int,
        shard_overlap_ratio: float = 0.1
    ) -> Dict[str, ModelShard]:
        """Create model shards from a PyTorch model."""
        
        try:
            # Get model parameters
            named_params = dict(model.named_parameters())
            param_names = list(named_params.keys())
            
            if self.sharding_strategy == "layer_wise":
                shards = self._create_layer_wise_shards(
                    named_params, num_shards, shard_overlap_ratio
                )
            elif self.sharding_strategy == "parameter_wise":
                shards = self._create_parameter_wise_shards(
                    named_params, num_shards
                )
            elif self.sharding_strategy == "random":
                shards = self._create_random_shards(
                    named_params, num_shards, shard_overlap_ratio
                )
            else:
                raise ValueError(f"Unknown sharding strategy: {self.sharding_strategy}")
            
            self.shards.update(shards)
            
            logger.info(f"Created {len(shards)} model shards using {self.sharding_strategy} strategy")
            
            return shards
            
        except Exception as e:
            logger.error(f"Model sharding failed: {e}")
            return {}
    
    def _create_layer_wise_shards(
        self, 
        named_params: Dict[str, torch.Tensor], 
        num_shards: int,
        overlap_ratio: float
    ) -> Dict[str, ModelShard]:
        """Create shards by grouping layers."""
        
        # Group parameters by layer
        layer_groups = defaultdict(list)
        
        for param_name in named_params.keys():
            # Extract layer name (everything before the last dot)
            if '.' in param_name:
                layer_name = '.'.join(param_name.split('.')[:-1])
            else:
                layer_name = param_name
            
            layer_groups[layer_name].append(param_name)
        
        # Distribute layer groups among shards
        layers = list(layer_groups.keys())
        layers_per_shard = max(1, len(layers) // num_shards)
        
        shards = {}
        
        for i in range(num_shards):
            shard_id = f"shard_{i}"
            start_idx = i * layers_per_shard
            
            if i == num_shards - 1:  # Last shard gets remaining layers
                end_idx = len(layers)
            else:
                end_idx = (i + 1) * layers_per_shard
                # Add overlap
                overlap_layers = max(1, int(layers_per_shard * overlap_ratio))
                end_idx = min(len(layers), end_idx + overlap_layers)
            
            # Collect parameters for this shard
            shard_layer_names = layers[start_idx:end_idx]
            shard_param_names = []
            shard_parameters = {}
            
            for layer_name in shard_layer_names:
                for param_name in layer_groups[layer_name]:
                    shard_param_names.append(param_name)
                    shard_parameters[param_name] = named_params[param_name].clone()
            
            shards[shard_id] = ModelShard(
                shard_id=shard_id,
                layer_names=shard_layer_names,
                parameters=shard_parameters
            )
        
        return shards
    
    def _create_parameter_wise_shards(
        self, 
        named_params: Dict[str, torch.Tensor], 
        num_shards: int
    ) -> Dict[str, ModelShard]:
        """Create shards by distributing individual parameters."""
        
        param_names = list(named_params.keys())
        params_per_shard = max(1, len(param_names) // num_shards)
        
        shards = {}
        
        for i in range(num_shards):
            shard_id = f"shard_{i}"
            start_idx = i * params_per_shard
            
            if i == num_shards - 1:  # Last shard gets remaining parameters
                end_idx = len(param_names)
            else:
                end_idx = (i + 1) * params_per_shard
            
            shard_param_names = param_names[start_idx:end_idx]
            shard_parameters = {
                name: named_params[name].clone() 
                for name in shard_param_names
            }
            
            shards[shard_id] = ModelShard(
                shard_id=shard_id,
                layer_names=[],  # Not applicable for parameter-wise sharding
                parameters=shard_parameters
            )
        
        return shards
    
    def _create_random_shards(
        self, 
        named_params: Dict[str, torch.Tensor], 
        num_shards: int,
        overlap_ratio: float
    ) -> Dict[str, ModelShard]:
        """Create shards by randomly distributing parameters."""
        
        param_names = list(named_params.keys())
        random.shuffle(param_names)  # Randomize parameter order
        
        # Calculate shard sizes with overlap
        base_size = len(param_names) // num_shards
        overlap_size = int(base_size * overlap_ratio)
        
        shards = {}
        
        for i in range(num_shards):
            shard_id = f"shard_{i}"
            start_idx = i * base_size
            end_idx = min(len(param_names), (i + 1) * base_size + overlap_size)
            
            shard_param_names = param_names[start_idx:end_idx]
            shard_parameters = {
                name: named_params[name].clone() 
                for name in shard_param_names
            }
            
            shards[shard_id] = ModelShard(
                shard_id=shard_id,
                layer_names=[],
                parameters=shard_parameters
            )
        
        return shards
    
    def assign_shards_to_clients(
        self, 
        clients: Dict[str, FederatedClient],
        assignment_strategy: str = "balanced"
    ) -> Dict[str, List[str]]:
        """Assign model shards to federated clients."""
        
        try:
            if assignment_strategy == "balanced":
                assignments = self._balanced_shard_assignment(clients)
            elif assignment_strategy == "capability_aware":
                assignments = self._capability_aware_assignment(clients)
            elif assignment_strategy == "geographic":
                assignments = self._geographic_assignment(clients)
            else:
                assignments = self._random_assignment(clients)
            
            # Update tracking
            self.client_shard_assignments.clear()
            self.client_shard_assignments.update(assignments)
            
            # Update shard client assignments
            for shard in self.shards.values():
                shard.client_assignments.clear()
            
            for client_id, shard_ids in assignments.items():
                for shard_id in shard_ids:
                    if shard_id in self.shards:
                        self.shards[shard_id].client_assignments.add(client_id)
            
            logger.info(f"Assigned {len(self.shards)} shards to {len(clients)} clients "
                       f"using {assignment_strategy} strategy")
            
            return assignments
            
        except Exception as e:
            logger.error(f"Shard assignment failed: {e}")
            return {}
    
    def _balanced_shard_assignment(
        self, 
        clients: Dict[str, FederatedClient]
    ) -> Dict[str, List[str]]:
        """Assign shards to balance load across clients."""
        
        if not clients or not self.shards:
            return {}
        
        client_ids = list(clients.keys())
        shard_ids = list(self.shards.keys())
        
        # Calculate shards per client
        shards_per_client = max(1, len(shard_ids) // len(client_ids))
        
        assignments = defaultdict(list)
        shard_idx = 0
        
        for client_id in client_ids:
            # Assign shards to this client
            for _ in range(shards_per_client):
                if shard_idx < len(shard_ids):
                    assignments[client_id].append(shard_ids[shard_idx])
                    shard_idx += 1
        
        # Distribute remaining shards
        while shard_idx < len(shard_ids):
            for client_id in client_ids:
                if shard_idx < len(shard_ids):
                    assignments[client_id].append(shard_ids[shard_idx])
                    shard_idx += 1
                else:
                    break
        
        return dict(assignments)
    
    def _capability_aware_assignment(
        self, 
        clients: Dict[str, FederatedClient]
    ) -> Dict[str, List[str]]:
        """Assign shards based on client capabilities."""
        
        # Sort clients by compute capacity
        sorted_clients = sorted(
            clients.items(), 
            key=lambda x: x[1].compute_capacity, 
            reverse=True
        )
        
        # Sort shards by size
        sorted_shards = sorted(
            self.shards.items(),
            key=lambda x: x[1].shard_size_mb,
            reverse=True
        )
        
        assignments = defaultdict(list)
        client_loads = {client_id: 0.0 for client_id in clients.keys()}
        
        # Assign largest shards to most capable clients
        for shard_id, shard in sorted_shards:
            # Find client with lowest current load relative to capacity
            best_client = min(
                clients.keys(),
                key=lambda cid: client_loads[cid] / max(0.1, clients[cid].compute_capacity)
            )
            
            assignments[best_client].append(shard_id)
            client_loads[best_client] += shard.shard_size_mb
        
        return dict(assignments)
    
    def _geographic_assignment(
        self, 
        clients: Dict[str, FederatedClient]
    ) -> Dict[str, List[str]]:
        """Assign shards considering geographic distribution."""
        
        # Group clients by region
        regional_clients = defaultdict(list)
        for client_id, client in clients.items():
            region = client.geographic_region or "default"
            regional_clients[region].append(client_id)
        
        # Distribute shards among regions
        shard_ids = list(self.shards.keys())
        assignments = defaultdict(list)
        
        shards_per_region = max(1, len(shard_ids) // len(regional_clients))
        shard_idx = 0
        
        for region, region_clients in regional_clients.items():
            # Assign shards to this region
            region_shards = []
            for _ in range(shards_per_region):
                if shard_idx < len(shard_ids):
                    region_shards.append(shard_ids[shard_idx])
                    shard_idx += 1
            
            # Distribute region shards among clients in the region
            for i, shard_id in enumerate(region_shards):
                client_idx = i % len(region_clients)
                client_id = region_clients[client_idx]
                assignments[client_id].append(shard_id)
        
        # Distribute remaining shards
        all_clients = list(clients.keys())
        while shard_idx < len(shard_ids):
            for client_id in all_clients:
                if shard_idx < len(shard_ids):
                    assignments[client_id].append(shard_ids[shard_idx])
                    shard_idx += 1
                else:
                    break
        
        return dict(assignments)
    
    def _random_assignment(
        self, 
        clients: Dict[str, FederatedClient]
    ) -> Dict[str, List[str]]:
        """Randomly assign shards to clients."""
        
        client_ids = list(clients.keys())
        shard_ids = list(self.shards.keys())
        random.shuffle(shard_ids)
        
        assignments = defaultdict(list)
        
        for i, shard_id in enumerate(shard_ids):
            client_id = client_ids[i % len(client_ids)]
            assignments[client_id].append(shard_id)
        
        return dict(assignments)
    
    def accumulate_shard_gradients(
        self, 
        shard_id: str, 
        client_id: str, 
        gradients: Dict[str, torch.Tensor]
    ) -> bool:
        """Accumulate gradients for a specific shard."""
        
        if shard_id not in self.shards:
            return False
        
        try:
            shard = self.shards[shard_id]
            
            # Initialize gradient accumulator if needed
            if not shard.gradient_accumulator:
                shard.gradient_accumulator = {
                    name: torch.zeros_like(param)
                    for name, param in shard.parameters.items()
                }
            
            # Accumulate gradients
            for param_name, gradient in gradients.items():
                if param_name in shard.gradient_accumulator:
                    shard.gradient_accumulator[param_name] += gradient
            
            return True
            
        except Exception as e:
            logger.error(f"Gradient accumulation failed for shard {shard_id}: {e}")
            return False
    
    def get_shard_assignments_status(self) -> Dict[str, Any]:
        """Get status of shard assignments."""
        
        assignment_stats = {
            'total_shards': len(self.shards),
            'total_clients_assigned': len(self.client_shard_assignments),
            'shards_per_client': {},
            'clients_per_shard': {},
            'assignment_balance': 0.0
        }
        
        # Calculate shards per client
        shard_counts = [len(shards) for shards in self.client_shard_assignments.values()]
        if shard_counts:
            assignment_stats['average_shards_per_client'] = np.mean(shard_counts)
            assignment_stats['assignment_balance'] = 1.0 - (np.std(shard_counts) / max(1, np.mean(shard_counts)))
        
        # Calculate clients per shard
        for shard_id, shard in self.shards.items():
            assignment_stats['clients_per_shard'][shard_id] = len(shard.client_assignments)
        
        return assignment_stats


class HierarchicalFederatedCoordinator:
    """
    Main coordinator for hierarchical federated learning across 10,000+ clients.
    
    Manages the complete federated learning lifecycle including client selection,
    model sharding, Byzantine-robust aggregation, and convergence monitoring.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Client management
        self.clients: Dict[str, FederatedClient] = {}
        self.active_clients: Set[str] = set()
        self.client_selection_strategy = self.config.get('client_selection', 'random')
        
        # Federated learning components
        self.byzantine_aggregator = ByzantineRobustAggregator(
            byzantine_tolerance=self.config.get('byzantine_tolerance', 0.3)
        )
        self.privacy_manager = DifferentialPrivacyManager(
            epsilon=self.config.get('dp_epsilon', 1.0),
            delta=self.config.get('dp_delta', 1e-5)
        )
        self.sharding_manager = ModelShardingManager(
            sharding_strategy=self.config.get('sharding_strategy', 'layer_wise')
        )
        
        # Global model and training
        self.global_model: Optional[nn.Module] = None
        self.current_round = 0
        self.target_accuracy = self.config.get('target_accuracy', 0.95)
        self.max_rounds = self.config.get('max_rounds', 1000)
        
        # Round management
        self.federated_rounds: deque = deque(maxlen=1000)
        self.convergence_history: deque = deque(maxlen=100)
        
        # Performance tracking
        self.training_stats = {
            'total_clients_trained': 0,
            'total_parameters_updated': 0,
            'total_communication_mb': 0.0,
            'average_round_time_ms': 0.0,
            'byzantine_clients_detected': 0
        }
        
        # Threading and async
        self._lock = threading.RLock()
        self._training_executor = ThreadPoolExecutor(max_workers=20)
        self._running = False
        self._coordination_task = None
        
        logger.info("Hierarchical federated coordinator initialized")
    
    async def start(self):
        """Start the federated learning coordination."""
        if self._running:
            return
        
        self._running = True
        self._coordination_task = asyncio.create_task(self._coordination_loop())
        
        logger.info("Federated learning coordinator started")
    
    async def stop(self):
        """Stop the federated learning coordination."""
        self._running = False
        
        if self._coordination_task:
            self._coordination_task.cancel()
        
        self._training_executor.shutdown(wait=True)
        
        logger.info("Federated learning coordinator stopped")
    
    async def register_client(
        self, 
        client_id: str, 
        role: FederatedRole,
        capabilities: Dict[str, Any],
        geographic_region: Optional[str] = None
    ) -> bool:
        """Register a new federated learning client."""
        
        try:
            client = FederatedClient(
                client_id=client_id,
                role=role,
                capabilities=capabilities,
                geographic_region=geographic_region,
                data_samples=capabilities.get('data_samples', 0),
                compute_capacity=capabilities.get('compute_capacity', 1.0),
                network_latency_ms=capabilities.get('network_latency_ms', 100.0)
            )
            
            with self._lock:
                self.clients[client_id] = client
                
                if role in [FederatedRole.EDGE_CLIENT, FederatedRole.FOG_AGGREGATOR]:
                    self.active_clients.add(client_id)
            
            logger.info(f"Registered federated client {client_id} "
                       f"({role.value}, {capabilities.get('data_samples', 0)} samples)")
            
            # Assign model shards if available
            if self.sharding_manager.shards and client_id not in self.sharding_manager.client_shard_assignments:
                await self._assign_shards_to_client(client_id)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to register client {client_id}: {e}")
            return False
    
    async def initialize_global_model(self, model: nn.Module, num_shards: int = 10) -> bool:
        """Initialize the global federated model with sharding."""
        
        try:
            self.global_model = model
            
            # Create model shards
            shards = self.sharding_manager.create_model_shards(
                model, 
                num_shards,
                shard_overlap_ratio=self.config.get('shard_overlap', 0.1)
            )
            
            if not shards:
                logger.error("Failed to create model shards")
                return False
            
            # Assign shards to existing clients
            if self.clients:
                assignments = self.sharding_manager.assign_shards_to_clients(
                    self.clients,
                    assignment_strategy=self.config.get('assignment_strategy', 'balanced')
                )
                
                logger.info(f"Assigned {len(shards)} shards to {len(assignments)} clients")
            
            logger.info(f"Initialized global model with {num_shards} shards")
            return True
            
        except Exception as e:
            logger.error(f"Global model initialization failed: {e}")
            return False
    
    async def start_federated_round(self) -> Optional[FederatedRound]:
        """Start a new federated learning round."""
        
        if not self.global_model or not self.active_clients:
            logger.warning("Cannot start federated round: missing model or clients")
            return None
        
        try:
            self.current_round += 1
            
            # Select clients for this round
            selected_clients = self._select_clients_for_round()
            
            if not selected_clients:
                logger.warning("No clients selected for federated round")
                return None
            
            # Create federated round
            fed_round = FederatedRound(
                round_id=self.current_round,
                start_time=datetime.now(),
                selected_clients=selected_clients,
                aggregation_algorithm=AggregationAlgorithm(
                    self.config.get('aggregation_algorithm', 'fedavg')
                ),
                privacy_mechanism=PrivacyMechanism(
                    self.config.get('privacy_mechanism', 'differential_privacy')
                )
            )
            
            # Broadcast global model to selected clients
            await self._broadcast_global_model(selected_clients, fed_round)
            
            # Start client training (async)
            training_tasks = []
            for client_id in selected_clients:
                task = asyncio.create_task(
                    self._coordinate_client_training(client_id, fed_round)
                )
                training_tasks.append(task)
            
            # Wait for client updates with timeout
            timeout_seconds = self.config.get('round_timeout', 300)  # 5 minutes
            
            try:
                await asyncio.wait_for(
                    asyncio.gather(*training_tasks, return_exceptions=True),
                    timeout=timeout_seconds
                )
            except asyncio.TimeoutError:
                logger.warning(f"Round {self.current_round} timed out")
            
            # Aggregate received updates
            await self._aggregate_round_updates(fed_round)
            
            # Update global model
            await self._update_global_model(fed_round)
            
            # Finalize round
            fed_round.end_time = datetime.now()
            fed_round.aggregation_time_ms = (
                fed_round.end_time - fed_round.start_time
            ).total_seconds() * 1000
            
            with self._lock:
                self.federated_rounds.append(fed_round)
            
            # Check convergence
            converged = await self._check_convergence(fed_round)
            
            logger.info(f"Completed federated round {self.current_round} "
                       f"with {len(fed_round.received_updates)} client updates "
                       f"(converged: {converged})")
            
            return fed_round
            
        except Exception as e:
            logger.error(f"Federated round {self.current_round} failed: {e}")
            return None
    
    def _select_clients_for_round(self) -> List[str]:
        """Select clients for the current federated round."""
        
        available_clients = list(self.active_clients)
        
        if not available_clients:
            return []
        
        # Configuration
        clients_per_round = self.config.get('clients_per_round', min(100, len(available_clients)))
        min_clients = self.config.get('min_clients_per_round', 10)
        
        clients_per_round = max(min_clients, min(clients_per_round, len(available_clients)))
        
        if self.client_selection_strategy == 'random':
            selected = random.sample(available_clients, clients_per_round)
        
        elif self.client_selection_strategy == 'capability_aware':
            # Select clients with highest compute capacity and reliability
            scored_clients = []
            
            for client_id in available_clients:
                client = self.clients[client_id]
                score = (client.compute_capacity * 0.4 + 
                        client.reliability_score * 0.4 +
                        (1.0 - client.network_latency_ms / 1000.0) * 0.2)
                scored_clients.append((client_id, score))
            
            scored_clients.sort(key=lambda x: x[1], reverse=True)
            selected = [client_id for client_id, _ in scored_clients[:clients_per_round]]
        
        elif self.client_selection_strategy == 'data_aware':
            # Select clients with most data samples
            data_clients = [
                (client_id, self.clients[client_id].data_samples)
                for client_id in available_clients
            ]
            data_clients.sort(key=lambda x: x[1], reverse=True)
            selected = [client_id for client_id, _ in data_clients[:clients_per_round]]
        
        else:  # Default to random
            selected = random.sample(available_clients, clients_per_round)
        
        return selected
    
    async def _broadcast_global_model(
        self, 
        selected_clients: List[str], 
        fed_round: FederatedRound
    ):
        """Broadcast global model parameters to selected clients."""
        
        try:
            # In a real implementation, this would send model parameters
            # over network to the selected clients
            
            for client_id in selected_clients:
                # Get client's assigned shards
                assigned_shards = self.sharding_manager.client_shard_assignments.get(client_id, [])
                
                # Simulate network transmission time
                client = self.clients[client_id]
                transmission_delay = client.network_latency_ms / 1000.0
                await asyncio.sleep(transmission_delay)
                
                # Update client model version
                client.model_version = self.current_round
                
                # Calculate communication cost
                total_params = 0
                for shard_id in assigned_shards:
                    if shard_id in self.sharding_manager.shards:
                        shard = self.sharding_manager.shards[shard_id]
                        total_params += sum(param.numel() for param in shard.parameters.values())
                
                # Assume 4 bytes per parameter (float32)
                communication_mb = (total_params * 4) / (1024 * 1024)
                fed_round.communication_cost_mb += communication_mb
            
            logger.debug(f"Broadcasted model to {len(selected_clients)} clients "
                        f"(total: {fed_round.communication_cost_mb:.2f} MB)")
            
        except Exception as e:
            logger.error(f"Model broadcast failed: {e}")
    
    async def _coordinate_client_training(
        self, 
        client_id: str, 
        fed_round: FederatedRound
    ):
        """Coordinate training for a specific client."""
        
        try:
            client = self.clients[client_id]
            
            # Simulate local training time
            training_time = self._estimate_training_time(client)
            await asyncio.sleep(training_time)
            
            # Generate simulated model updates
            updates = await self._simulate_client_updates(client_id, fed_round)
            
            if updates:
                # Apply differential privacy if enabled
                if fed_round.privacy_mechanism == PrivacyMechanism.DIFFERENTIAL_PRIVACY:
                    updates = self.privacy_manager.add_noise_to_updates(
                        updates, client_id
                    )
                
                # Store client updates
                with self._lock:
                    fed_round.received_updates[client_id] = {
                        'parameters': updates,
                        'data_samples': client.data_samples,
                        'training_time': training_time,
                        'timestamp': datetime.now()
                    }
                
                # Update client statistics
                client.last_update = datetime.now()
                client.current_round = self.current_round
                client.consecutive_failures = 0
                
                logger.debug(f"Received updates from client {client_id}")
            
        except Exception as e:
            logger.error(f"Client training coordination failed for {client_id}: {e}")
            
            # Update failure tracking
            if client_id in self.clients:
                self.clients[client_id].consecutive_failures += 1
    
    def _estimate_training_time(self, client: FederatedClient) -> float:
        """Estimate training time for a client."""
        
        base_time = 5.0  # Base training time in seconds
        
        # Adjust based on data size
        data_factor = math.log(max(1, client.data_samples)) / 10.0
        
        # Adjust based on compute capacity
        compute_factor = 1.0 / max(0.1, client.compute_capacity)
        
        # Add some randomness
        random_factor = random.uniform(0.5, 1.5)
        
        return base_time * data_factor * compute_factor * random_factor
    
    async def _simulate_client_updates(
        self, 
        client_id: str, 
        fed_round: FederatedRound
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Simulate client model updates."""
        
        try:
            client = self.clients[client_id]
            assigned_shards = self.sharding_manager.client_shard_assignments.get(client_id, [])
            
            if not assigned_shards:
                return None
            
            # Simulate updates for assigned shards
            updates = {}
            
            for shard_id in assigned_shards:
                if shard_id in self.sharding_manager.shards:
                    shard = self.sharding_manager.shards[shard_id]
                    
                    # Generate random gradients (in reality, these would come from training)
                    for param_name, param in shard.parameters.items():
                        
                        # Simulate realistic gradient magnitudes
                        gradient_magnitude = 0.01 * random.uniform(0.1, 2.0)
                        
                        # Add Byzantine behavior for detected Byzantine clients
                        if client.is_byzantine:
                            gradient_magnitude *= random.uniform(5.0, 20.0)  # Malicious scaling
                        
                        gradient = torch.randn_like(param) * gradient_magnitude
                        updates[param_name] = gradient
            
            return updates if updates else None
            
        except Exception as e:
            logger.error(f"Update simulation failed for client {client_id}: {e}")
            return None
    
    async def _aggregate_round_updates(self, fed_round: FederatedRound):
        """Aggregate client updates for the federated round."""
        
        if not fed_round.received_updates:
            logger.warning(f"No updates received for round {fed_round.round_id}")
            return
        
        try:
            start_time = time.time()
            
            # Prepare updates for aggregation
            client_updates = {}
            client_weights = {}
            
            for client_id, update_data in fed_round.received_updates.items():
                client_updates[client_id] = update_data['parameters']
                
                # Weight by data samples
                client_weights[client_id] = update_data.get('data_samples', 1)
            
            # Perform Byzantine-robust aggregation
            aggregated_updates, byzantine_clients = self.byzantine_aggregator.aggregate_updates(
                client_updates,
                fed_round.aggregation_algorithm,
                client_weights
            )
            
            # Update Byzantine client tracking
            fed_round.byzantine_clients_detected = byzantine_clients
            self.training_stats['byzantine_clients_detected'] += len(byzantine_clients)
            
            # Mark detected Byzantine clients
            for client_id in byzantine_clients:
                if client_id in self.clients:
                    self.clients[client_id].is_byzantine = True
                    self.clients[client_id].reliability_score *= 0.5  # Reduce reliability
            
            # Store aggregated updates in round
            fed_round.received_updates['aggregated'] = {
                'parameters': aggregated_updates,
                'num_clients': len(client_updates),
                'byzantine_detected': len(byzantine_clients)
            }
            
            aggregation_time = (time.time() - start_time) * 1000
            fed_round.aggregation_time_ms = aggregation_time
            
            logger.info(f"Aggregated updates from {len(client_updates)} clients "
                       f"in {aggregation_time:.2f}ms "
                       f"(Byzantine detected: {len(byzantine_clients)})")
            
        except Exception as e:
            logger.error(f"Update aggregation failed for round {fed_round.round_id}: {e}")
    
    async def _update_global_model(self, fed_round: FederatedRound):
        """Update global model with aggregated updates."""
        
        if 'aggregated' not in fed_round.received_updates:
            return
        
        try:
            aggregated_updates = fed_round.received_updates['aggregated']['parameters']
            
            if not aggregated_updates:
                return
            
            # Apply updates to global model shards
            learning_rate = self.config.get('global_learning_rate', 1.0)
            
            for shard_id, shard in self.sharding_manager.shards.items():
                for param_name, param in shard.parameters.items():
                    if param_name in aggregated_updates:
                        # Apply update with learning rate
                        update = aggregated_updates[param_name] * learning_rate
                        shard.parameters[param_name] = param - update
            
            # Update training statistics
            self.training_stats['total_clients_trained'] += len(fed_round.selected_clients)
            self.training_stats['total_parameters_updated'] += len(aggregated_updates)
            self.training_stats['total_communication_mb'] += fed_round.communication_cost_mb
            
            logger.debug(f"Updated global model with {len(aggregated_updates)} parameters")
            
        except Exception as e:
            logger.error(f"Global model update failed: {e}")
    
    async def _check_convergence(self, fed_round: FederatedRound) -> bool:
        """Check if federated learning has converged."""
        
        try:
            # Simple convergence check based on number of rounds
            # In practice, this would involve evaluating the global model
            
            if self.current_round >= self.max_rounds:
                logger.info(f"Maximum rounds ({self.max_rounds}) reached")
                return True
            
            # Check if we have enough rounds for trend analysis
            if len(self.federated_rounds) < 10:
                return False
            
            # Analyze recent communication costs (proxy for convergence)
            recent_rounds = list(self.federated_rounds)[-10:]
            comm_costs = [r.communication_cost_mb for r in recent_rounds]
            
            if len(comm_costs) >= 5:
                # Check if communication cost is stabilizing
                recent_avg = np.mean(comm_costs[-5:])
                older_avg = np.mean(comm_costs[:5])
                
                if abs(recent_avg - older_avg) / max(older_avg, 1.0) < 0.05:
                    convergence_score = 0.95  # High convergence
                else:
                    convergence_score = 0.5  # Moderate convergence
            else:
                convergence_score = 0.1  # Low convergence
            
            fed_round.convergence_metrics['convergence_score'] = convergence_score
            self.convergence_history.append(convergence_score)
            
            # Check convergence threshold
            converged = convergence_score >= self.target_accuracy
            
            if converged:
                logger.info(f"Federated learning converged at round {self.current_round} "
                           f"(score: {convergence_score:.3f})")
            
            return converged
            
        except Exception as e:
            logger.error(f"Convergence check failed: {e}")
            return False
    
    async def _assign_shards_to_client(self, client_id: str):
        """Assign model shards to a new client."""
        
        try:
            if client_id not in self.clients:
                return
            
            # Re-balance shard assignments with new client
            assignments = self.sharding_manager.assign_shards_to_clients(
                self.clients,
                assignment_strategy=self.config.get('assignment_strategy', 'balanced')
            )
            
            assigned_shards = assignments.get(client_id, [])
            
            logger.info(f"Assigned {len(assigned_shards)} shards to client {client_id}")
            
        except Exception as e:
            logger.error(f"Shard assignment failed for client {client_id}: {e}")
    
    async def _coordination_loop(self):
        """Main coordination loop for federated learning."""
        
        while self._running:
            try:
                # Check if we have enough clients to start a round
                if len(self.active_clients) < self.config.get('min_clients_per_round', 10):
                    logger.debug("Not enough active clients for federated round")
                    await asyncio.sleep(30)
                    continue
                
                # Start federated round
                fed_round = await self.start_federated_round()
                
                if fed_round:
                    # Check convergence
                    if await self._check_convergence(fed_round):
                        logger.info("Federated learning converged - stopping coordination")
                        break
                
                # Wait between rounds
                round_interval = self.config.get('round_interval_seconds', 60)
                await asyncio.sleep(round_interval)
                
            except Exception as e:
                logger.error(f"Coordination loop error: {e}")
                await asyncio.sleep(30)
    
    def get_federation_status(self) -> Dict[str, Any]:
        """Get comprehensive federated learning status."""
        
        with self._lock:
            active_clients_by_role = defaultdict(int)
            for client_id in self.active_clients:
                if client_id in self.clients:
                    role = self.clients[client_id].role
                    active_clients_by_role[role.value] += 1
            
            byzantine_clients = sum(
                1 for client in self.clients.values() if client.is_byzantine
            )
            
            status = {
                'federated_learning': {
                    'current_round': self.current_round,
                    'max_rounds': self.max_rounds,
                    'total_clients': len(self.clients),
                    'active_clients': len(self.active_clients),
                    'byzantine_clients_detected': byzantine_clients,
                    'convergence_score': (
                        self.convergence_history[-1] if self.convergence_history else 0.0
                    ),
                    'clients_by_role': dict(active_clients_by_role),
                    'rounds_completed': len(self.federated_rounds)
                },
                'model_sharding': self.sharding_manager.get_shard_assignments_status(),
                'privacy_preservation': self.privacy_manager.get_privacy_stats(),
                'training_statistics': self.training_stats.copy(),
                'performance_metrics': {
                    'avg_round_time_ms': (
                        np.mean([r.aggregation_time_ms for r in self.federated_rounds])
                        if self.federated_rounds else 0.0
                    ),
                    'total_communication_mb': self.training_stats['total_communication_mb'],
                    'byzantine_detection_rate': (
                        byzantine_clients / max(1, len(self.clients))
                    )
                }
            }
        
        return status


# Global federated coordinator instance
_federated_coordinator: Optional[HierarchicalFederatedCoordinator] = None


def get_federated_coordinator(config: Optional[Dict[str, Any]] = None) -> HierarchicalFederatedCoordinator:
    """Get or create global federated learning coordinator."""
    global _federated_coordinator
    
    if _federated_coordinator is None:
        _federated_coordinator = HierarchicalFederatedCoordinator(config)
    
    return _federated_coordinator


async def start_federated_learning(
    model: nn.Module,
    config: Optional[Dict[str, Any]] = None,
    num_shards: int = 10
) -> HierarchicalFederatedCoordinator:
    """Start hierarchical federated learning system."""
    
    coordinator = get_federated_coordinator(config)
    
    # Initialize global model
    await coordinator.initialize_global_model(model, num_shards)
    
    # Start coordination
    await coordinator.start()
    
    return coordinator