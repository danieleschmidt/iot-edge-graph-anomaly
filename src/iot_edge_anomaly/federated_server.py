"""
Federated Learning Server for IoT Edge Anomaly Detection.

Implements a Byzantine-robust federated aggregation server that coordinates
multiple edge devices for collaborative anomaly detection learning without
sharing raw sensor data.

Key Features:
- Byzantine-robust aggregation algorithms
- Differential privacy mechanisms
- Adaptive client selection
- Real-time model performance monitoring
- Secure model aggregation with blockchain readiness
"""

import asyncio
import logging
import time
import json
import hashlib
import uuid
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .models.lstm_autoencoder import LSTMAutoencoder
from .models.federated_learning import FederatedLearningClient, ModelUpdate, FederatedLearningConfig
from .security.advanced_security import AdvancedSecurityFramework
from .monitoring.advanced_metrics import AdvancedMetricsCollector

logger = logging.getLogger(__name__)


@dataclass
class ClientInfo:
    """Information about a federated learning client."""
    client_id: str
    last_seen: datetime
    model_version: int
    data_quality_score: float
    contribution_score: float
    is_byzantine: bool = False
    reputation_score: float = 1.0
    
    
@dataclass
class AggregationRound:
    """Information about a federated aggregation round."""
    round_id: int
    start_time: datetime
    end_time: Optional[datetime]
    participating_clients: List[str]
    global_model_version: int
    aggregation_method: str
    convergence_score: float = 0.0
    byzantine_clients_detected: List[str] = None
    
    def __post_init__(self):
        if self.byzantine_clients_detected is None:
            self.byzantine_clients_detected = []


class ByzantineRobustAggregator:
    """
    Byzantine-robust aggregation algorithms for federated learning.
    
    Implements multiple aggregation strategies to handle malicious clients
    and ensure model convergence in adversarial environments.
    """
    
    def __init__(self, byzantine_tolerance: float = 0.3):
        self.byzantine_tolerance = byzantine_tolerance  # Max fraction of Byzantine clients
        
    def aggregate_krum(self, model_updates: List[ModelUpdate], k: int = None) -> torch.Tensor:
        """
        Krum aggregation: Select k closest models and average them.
        
        Args:
            model_updates: List of model updates from clients
            k: Number of models to select (default: n - f - 2 where f is Byzantine bound)
            
        Returns:
            Aggregated model weights
        """
        n = len(model_updates)
        f = int(n * self.byzantine_tolerance)  # Byzantine bound
        k = k or (n - f - 2)
        
        if k <= 0:
            raise ValueError(f"Invalid k={k} for n={n} clients with Byzantine tolerance {self.byzantine_tolerance}")
        
        # Extract weight vectors
        weight_vectors = []
        for update in model_updates:
            weights = []
            for param in update.model_weights.values():
                weights.extend(param.flatten().tolist())
            weight_vectors.append(np.array(weights))
        
        # Compute pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                distances[i, j] = np.linalg.norm(weight_vectors[i] - weight_vectors[j])
        
        # For each client, find sum of distances to k nearest neighbors
        krum_scores = []
        for i in range(n):
            sorted_distances = np.sort(distances[i])
            krum_score = np.sum(sorted_distances[1:k+1])  # Exclude distance to self
            krum_scores.append(krum_score)
        
        # Select client with minimum Krum score
        selected_client_idx = np.argmin(krum_scores)
        
        logger.info(f"Krum aggregation selected client {model_updates[selected_client_idx].client_id}")
        
        return self._convert_weights_dict_to_tensor(model_updates[selected_client_idx].model_weights)
    
    def aggregate_trimmed_mean(self, model_updates: List[ModelUpdate], trim_ratio: float = 0.2) -> torch.Tensor:
        """
        Trimmed mean aggregation: Remove outlier models and average the rest.
        
        Args:
            model_updates: List of model updates from clients
            trim_ratio: Fraction of outliers to remove from each side
            
        Returns:
            Aggregated model weights
        """
        n = len(model_updates)
        trim_count = int(n * trim_ratio)
        
        # Convert all model weights to vectors
        weight_vectors = []
        for update in model_updates:
            weights = []
            for param in update.model_weights.values():
                weights.extend(param.flatten().tolist())
            weight_vectors.append(np.array(weights))
        
        weight_matrix = np.array(weight_vectors)  # [n_clients, n_params]
        
        # For each parameter, compute trimmed mean
        trimmed_weights = []
        for param_idx in range(weight_matrix.shape[1]):
            param_values = weight_matrix[:, param_idx]
            sorted_values = np.sort(param_values)
            
            # Remove outliers from both ends
            if trim_count > 0:
                trimmed_values = sorted_values[trim_count:-trim_count]
            else:
                trimmed_values = sorted_values
            
            trimmed_mean = np.mean(trimmed_values) if len(trimmed_values) > 0 else np.mean(param_values)
            trimmed_weights.append(trimmed_mean)
        
        # Convert back to model weight format
        return self._convert_vector_to_weights_dict(
            np.array(trimmed_weights), 
            model_updates[0].model_weights
        )
    
    def aggregate_median(self, model_updates: List[ModelUpdate]) -> torch.Tensor:
        """
        Coordinate-wise median aggregation.
        
        Args:
            model_updates: List of model updates from clients
            
        Returns:
            Aggregated model weights
        """
        # Convert all model weights to vectors
        weight_vectors = []
        for update in model_updates:
            weights = []
            for param in update.model_weights.values():
                weights.extend(param.flatten().tolist())
            weight_vectors.append(np.array(weights))
        
        weight_matrix = np.array(weight_vectors)  # [n_clients, n_params]
        
        # Compute coordinate-wise median
        median_weights = np.median(weight_matrix, axis=0)
        
        return self._convert_vector_to_weights_dict(
            median_weights, 
            model_updates[0].model_weights
        )
    
    def _convert_weights_dict_to_tensor(self, weights_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert model weights dictionary to tensor format."""
        return weights_dict
    
    def _convert_vector_to_weights_dict(self, weight_vector: np.ndarray, template_weights: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Convert weight vector back to model weights dictionary format."""
        result = {}
        idx = 0
        
        for param_name, param_tensor in template_weights.items():
            param_size = param_tensor.numel()
            param_weights = weight_vector[idx:idx + param_size]
            result[param_name] = torch.tensor(param_weights.reshape(param_tensor.shape), dtype=param_tensor.dtype)
            idx += param_size
        
        return result


class DifferentialPrivacyMechanism:
    """
    Differential privacy mechanisms for federated learning.
    
    Provides privacy protection for client model updates through
    noise injection and gradient clipping.
    """
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5, sensitivity: float = 1.0):
        self.epsilon = epsilon  # Privacy budget
        self.delta = delta     # Failure probability
        self.sensitivity = sensitivity  # L2 sensitivity
        
    def add_gaussian_noise(self, model_update: ModelUpdate) -> ModelUpdate:
        """
        Add Gaussian noise to model weights for differential privacy.
        
        Args:
            model_update: Original model update
            
        Returns:
            Noisy model update
        """
        # Calculate noise scale for Gaussian mechanism
        noise_scale = self.sensitivity * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
        
        noisy_weights = {}
        for param_name, param_tensor in model_update.model_weights.items():
            # Add Gaussian noise
            noise = torch.normal(0, noise_scale, size=param_tensor.shape)
            noisy_weights[param_name] = param_tensor + noise
        
        # Create new model update with noisy weights
        noisy_update = ModelUpdate(
            client_id=model_update.client_id,
            model_weights=noisy_weights,
            num_samples=model_update.num_samples,
            loss=model_update.loss,
            accuracy=model_update.accuracy,
            timestamp=model_update.timestamp
        )
        
        return noisy_update
    
    def clip_gradients(self, model_update: ModelUpdate, clip_norm: float = 1.0) -> ModelUpdate:
        """
        Clip gradients to bound their L2 norm.
        
        Args:
            model_update: Model update to clip
            clip_norm: Maximum L2 norm for gradients
            
        Returns:
            Clipped model update
        """
        # Calculate total gradient norm
        total_norm = 0.0
        for param_tensor in model_update.model_weights.values():
            total_norm += param_tensor.norm().item() ** 2
        total_norm = total_norm ** 0.5
        
        # Clip if necessary
        if total_norm > clip_norm:
            clip_factor = clip_norm / total_norm
            clipped_weights = {}
            for param_name, param_tensor in model_update.model_weights.items():
                clipped_weights[param_name] = param_tensor * clip_factor
        else:
            clipped_weights = model_update.model_weights
        
        clipped_update = ModelUpdate(
            client_id=model_update.client_id,
            model_weights=clipped_weights,
            num_samples=model_update.num_samples,
            loss=model_update.loss,
            accuracy=model_update.accuracy,
            timestamp=model_update.timestamp
        )
        
        return clipped_update


class FederatedLearningServer:
    """
    Federated Learning Server for IoT Anomaly Detection.
    
    Coordinates multiple edge devices for collaborative learning with
    Byzantine robustness and differential privacy.
    """
    
    def __init__(
        self,
        config: FederatedLearningConfig,
        aggregation_method: str = "byzantine_robust",
        min_clients: int = 3,
        max_clients: int = 100
    ):
        self.config = config
        self.aggregation_method = aggregation_method
        self.min_clients = min_clients
        self.max_clients = max_clients
        
        # Server state
        self.global_model = None
        self.global_model_version = 0
        self.current_round = 0
        self.clients: Dict[str, ClientInfo] = {}
        self.aggregation_history: List[AggregationRound] = []
        
        # Security and privacy
        self.security_framework = AdvancedSecurityFramework()
        self.dp_mechanism = DifferentialPrivacyMechanism(
            epsilon=config.privacy_epsilon,
            delta=config.privacy_delta
        )
        self.aggregator = ByzantineRobustAggregator(
            byzantine_tolerance=config.byzantine_tolerance
        )
        
        # Monitoring
        self.metrics_collector = AdvancedMetricsCollector()
        
        # Server configuration
        self.server_id = str(uuid.uuid4())
        self.start_time = datetime.now()
        
        logger.info(f"Initialized Federated Learning Server {self.server_id}")
        logger.info(f"Aggregation method: {aggregation_method}")
        logger.info(f"Min clients: {min_clients}, Max clients: {max_clients}")
    
    def initialize_global_model(self, model_config: Dict[str, Any]) -> None:
        """
        Initialize the global model.
        
        Args:
            model_config: Configuration for the global model
        """
        self.global_model = LSTMAutoencoder(
            input_size=model_config.get('input_size', 5),
            hidden_size=model_config.get('hidden_size', 64),
            num_layers=model_config.get('num_layers', 2),
            dropout=model_config.get('dropout', 0.1)
        )
        
        self.global_model_version = 1
        logger.info(f"Initialized global model version {self.global_model_version}")
    
    def register_client(self, client_id: str, client_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register a new federated learning client.
        
        Args:
            client_id: Unique identifier for the client
            client_info: Client metadata and capabilities
            
        Returns:
            Registration response with global model
        """
        if client_id in self.clients:
            logger.warning(f"Client {client_id} already registered, updating info")
        
        # Create client info
        self.clients[client_id] = ClientInfo(
            client_id=client_id,
            last_seen=datetime.now(),
            model_version=0,
            data_quality_score=client_info.get('data_quality_score', 1.0),
            contribution_score=0.0,
            reputation_score=1.0
        )
        
        logger.info(f"Registered client {client_id}")
        
        # Return global model for initialization
        response = {
            'status': 'registered',
            'global_model_version': self.global_model_version,
            'global_model_weights': self._get_global_model_weights(),
            'server_config': {
                'aggregation_method': self.aggregation_method,
                'privacy_epsilon': self.config.privacy_epsilon,
                'byzantine_tolerance': self.config.byzantine_tolerance
            }
        }
        
        return response
    
    async def receive_model_update(self, model_update: ModelUpdate) -> Dict[str, Any]:
        """
        Receive and process a model update from a client.
        
        Args:
            model_update: Model update from client
            
        Returns:
            Response with update status
        """
        client_id = model_update.client_id
        
        # Validate client
        if client_id not in self.clients:
            logger.error(f"Received update from unregistered client {client_id}")
            return {'status': 'error', 'message': 'Client not registered'}
        
        # Security validation
        is_valid = await self._validate_model_update(model_update)
        if not is_valid:
            logger.warning(f"Invalid model update from client {client_id}")
            self.clients[client_id].is_byzantine = True
            return {'status': 'rejected', 'message': 'Invalid model update'}
        
        # Apply differential privacy
        if self.config.enable_differential_privacy:
            model_update = self.dp_mechanism.clip_gradients(model_update)
            model_update = self.dp_mechanism.add_gaussian_noise(model_update)
        
        # Update client info
        self.clients[client_id].last_seen = datetime.now()
        self.clients[client_id].model_version = model_update.timestamp
        
        # Store update for aggregation
        await self._store_model_update(model_update)
        
        # Record metrics
        await self.metrics_collector.record_federated_update(
            client_id=client_id,
            num_samples=model_update.num_samples,
            loss=model_update.loss,
            accuracy=model_update.accuracy
        )
        
        logger.debug(f"Received model update from client {client_id}")
        
        return {
            'status': 'received',
            'global_model_version': self.global_model_version,
            'next_aggregation_time': self._get_next_aggregation_time()
        }
    
    async def run_aggregation_round(self, selected_clients: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Run a federated aggregation round.
        
        Args:
            selected_clients: Optional list of specific clients to include
            
        Returns:
            Aggregation results
        """
        self.current_round += 1
        start_time = datetime.now()
        
        logger.info(f"Starting aggregation round {self.current_round}")
        
        # Select participating clients
        if selected_clients is None:
            selected_clients = self._select_clients_for_aggregation()
        
        if len(selected_clients) < self.min_clients:
            logger.warning(f"Insufficient clients for aggregation: {len(selected_clients)} < {self.min_clients}")
            return {'status': 'insufficient_clients', 'available_clients': len(selected_clients)}
        
        # Collect model updates
        model_updates = await self._collect_model_updates(selected_clients)
        
        if len(model_updates) < self.min_clients:
            logger.warning(f"Insufficient model updates: {len(model_updates)} < {self.min_clients}")
            return {'status': 'insufficient_updates', 'received_updates': len(model_updates)}
        
        # Detect Byzantine clients
        byzantine_clients = self._detect_byzantine_clients(model_updates)
        
        # Filter out Byzantine updates
        valid_updates = [update for update in model_updates if update.client_id not in byzantine_clients]
        
        if len(valid_updates) < self.min_clients:
            logger.error(f"Too many Byzantine clients detected: {len(byzantine_clients)}")
            return {'status': 'too_many_byzantine', 'byzantine_clients': byzantine_clients}
        
        # Aggregate models
        aggregated_weights = await self._aggregate_models(valid_updates)
        
        # Update global model
        self._update_global_model(aggregated_weights)
        
        # Update client reputation scores
        self._update_client_reputation(model_updates, byzantine_clients)
        
        # Record aggregation round
        end_time = datetime.now()
        aggregation_round = AggregationRound(
            round_id=self.current_round,
            start_time=start_time,
            end_time=end_time,
            participating_clients=selected_clients,
            global_model_version=self.global_model_version,
            aggregation_method=self.aggregation_method,
            byzantine_clients_detected=byzantine_clients
        )
        
        self.aggregation_history.append(aggregation_round)
        
        # Calculate convergence metrics
        convergence_score = self._calculate_convergence_score(model_updates)
        aggregation_round.convergence_score = convergence_score
        
        logger.info(f"Completed aggregation round {self.current_round}")
        logger.info(f"Participating clients: {len(selected_clients)}")
        logger.info(f"Byzantine clients detected: {len(byzantine_clients)}")
        logger.info(f"Convergence score: {convergence_score:.4f}")
        
        return {
            'status': 'completed',
            'round_id': self.current_round,
            'global_model_version': self.global_model_version,
            'participating_clients': len(selected_clients),
            'byzantine_clients_detected': len(byzantine_clients),
            'convergence_score': convergence_score,
            'aggregation_time_seconds': (end_time - start_time).total_seconds()
        }
    
    async def _validate_model_update(self, model_update: ModelUpdate) -> bool:
        """Validate model update for security and integrity."""
        try:
            # Check model structure
            if not model_update.model_weights:
                return False
            
            # Check for NaN or infinite values
            for param_tensor in model_update.model_weights.values():
                if torch.isnan(param_tensor).any() or torch.isinf(param_tensor).any():
                    return False
            
            # Check update size bounds
            total_params = sum(p.numel() for p in model_update.model_weights.values())
            if total_params > 1e7:  # Reasonable upper bound
                return False
            
            # Additional security checks via security framework
            threat_detected = await self.security_framework.detect_model_threats(model_update.model_weights)
            return not threat_detected
            
        except Exception as e:
            logger.error(f"Error validating model update: {e}")
            return False
    
    async def _store_model_update(self, model_update: ModelUpdate) -> None:
        """Store model update for aggregation."""
        # In a production system, this would store updates in a database
        # For now, we'll store in memory (implement as needed)
        pass
    
    def _select_clients_for_aggregation(self) -> List[str]:
        """Select clients for the next aggregation round."""
        # Filter active, non-Byzantine clients
        eligible_clients = []
        current_time = datetime.now()
        
        for client_id, client_info in self.clients.items():
            # Check if client is recently active
            time_since_last_seen = current_time - client_info.last_seen
            if time_since_last_seen < timedelta(minutes=10):  # 10 minute timeout
                if not client_info.is_byzantine and client_info.reputation_score > 0.5:
                    eligible_clients.append((client_id, client_info.contribution_score))
        
        # Sort by contribution score and select top clients
        eligible_clients.sort(key=lambda x: x[1], reverse=True)
        selected_clients = [client_id for client_id, _ in eligible_clients[:self.max_clients]]
        
        return selected_clients
    
    async def _collect_model_updates(self, selected_clients: List[str]) -> List[ModelUpdate]:
        """Collect model updates from selected clients."""
        # In a real implementation, this would fetch stored updates
        # For now, return empty list (implement as needed)
        return []
    
    def _detect_byzantine_clients(self, model_updates: List[ModelUpdate]) -> List[str]:
        """Detect Byzantine clients using statistical analysis."""
        if len(model_updates) < 3:
            return []
        
        byzantine_clients = []
        
        # Statistical outlier detection
        losses = [update.loss for update in model_updates]
        loss_mean = np.mean(losses)
        loss_std = np.std(losses)
        
        # Clients with losses more than 3 standard deviations away
        for update in model_updates:
            if abs(update.loss - loss_mean) > 3 * loss_std:
                byzantine_clients.append(update.client_id)
                logger.warning(f"Detected potential Byzantine client {update.client_id} with outlier loss {update.loss}")
        
        return byzantine_clients
    
    async def _aggregate_models(self, model_updates: List[ModelUpdate]) -> Dict[str, torch.Tensor]:
        """Aggregate model updates using the configured method."""
        if self.aggregation_method == "byzantine_robust":
            return self.aggregator.aggregate_krum(model_updates)
        elif self.aggregation_method == "trimmed_mean":
            return self.aggregator.aggregate_trimmed_mean(model_updates)
        elif self.aggregation_method == "median":
            return self.aggregator.aggregate_median(model_updates)
        else:
            # Default: FedAvg (weighted average)
            return self._federated_averaging(model_updates)
    
    def _federated_averaging(self, model_updates: List[ModelUpdate]) -> Dict[str, torch.Tensor]:
        """Standard FedAvg aggregation."""
        total_samples = sum(update.num_samples for update in model_updates)
        
        # Initialize aggregated weights
        aggregated_weights = {}
        first_update = model_updates[0]
        
        for param_name in first_update.model_weights:
            aggregated_weights[param_name] = torch.zeros_like(first_update.model_weights[param_name])
        
        # Weighted average
        for update in model_updates:
            weight = update.num_samples / total_samples
            for param_name, param_tensor in update.model_weights.items():
                aggregated_weights[param_name] += weight * param_tensor
        
        return aggregated_weights
    
    def _update_global_model(self, aggregated_weights: Dict[str, torch.Tensor]) -> None:
        """Update the global model with aggregated weights."""
        if self.global_model is not None:
            # Load aggregated weights into global model
            self.global_model.load_state_dict(aggregated_weights, strict=False)
            self.global_model_version += 1
        
    def _update_client_reputation(self, model_updates: List[ModelUpdate], byzantine_clients: List[str]) -> None:
        """Update client reputation scores based on behavior."""
        for update in model_updates:
            client_id = update.client_id
            if client_id in self.clients:
                if client_id in byzantine_clients:
                    # Decrease reputation for Byzantine behavior
                    self.clients[client_id].reputation_score *= 0.8
                    self.clients[client_id].is_byzantine = True
                else:
                    # Increase reputation for good behavior
                    self.clients[client_id].reputation_score = min(1.0, self.clients[client_id].reputation_score * 1.05)
                    self.clients[client_id].contribution_score += 1.0
    
    def _calculate_convergence_score(self, model_updates: List[ModelUpdate]) -> float:
        """Calculate convergence score for the aggregation round."""
        if len(model_updates) < 2:
            return 0.0
        
        # Calculate variance in losses as convergence metric
        losses = [update.loss for update in model_updates]
        loss_variance = np.var(losses)
        
        # Convert to convergence score (lower variance = higher convergence)
        convergence_score = 1.0 / (1.0 + loss_variance)
        
        return convergence_score
    
    def _get_global_model_weights(self) -> Dict[str, torch.Tensor]:
        """Get current global model weights."""
        if self.global_model is not None:
            return self.global_model.state_dict()
        return {}
    
    def _get_next_aggregation_time(self) -> str:
        """Get estimated time for next aggregation round."""
        # Simple estimation: next aggregation in 5 minutes
        next_time = datetime.now() + timedelta(minutes=5)
        return next_time.isoformat()
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get comprehensive server status."""
        current_time = datetime.now()
        uptime = current_time - self.start_time
        
        active_clients = [
            client_id for client_id, client_info in self.clients.items()
            if (current_time - client_info.last_seen) < timedelta(minutes=10)
        ]
        
        return {
            'server_id': self.server_id,
            'uptime_seconds': uptime.total_seconds(),
            'current_round': self.current_round,
            'global_model_version': self.global_model_version,
            'total_clients': len(self.clients),
            'active_clients': len(active_clients),
            'aggregation_method': self.aggregation_method,
            'privacy_enabled': self.config.enable_differential_privacy,
            'last_aggregation': self.aggregation_history[-1].start_time.isoformat() if self.aggregation_history else None
        }


async def main():
    """Main entry point for federated learning server."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Federated Learning Server for IoT Anomaly Detection')
    parser.add_argument('--config', type=str, default='config/federated_server.yaml',
                       help='Configuration file path')
    parser.add_argument('--aggregation-method', type=str, default='byzantine_robust',
                       choices=['byzantine_robust', 'trimmed_mean', 'median', 'fedavg'],
                       help='Aggregation method')
    parser.add_argument('--min-clients', type=int, default=3,
                       help='Minimum number of clients for aggregation')
    parser.add_argument('--max-clients', type=int, default=100,
                       help='Maximum number of clients per round')
    parser.add_argument('--port', type=int, default=8080,
                       help='Server port')
    
    args = parser.parse_args()
    
    # Load configuration
    config = FederatedLearningConfig()
    
    # Create server
    server = FederatedLearningServer(
        config=config,
        aggregation_method=args.aggregation_method,
        min_clients=args.min_clients,
        max_clients=args.max_clients
    )
    
    # Initialize global model
    model_config = {
        'input_size': 5,
        'hidden_size': 64,
        'num_layers': 2,
        'dropout': 0.1
    }
    server.initialize_global_model(model_config)
    
    logger.info(f"Federated Learning Server started on port {args.port}")
    
    # In a real implementation, this would start an HTTP/gRPC server
    # For now, just log the server status
    while True:
        status = server.get_server_status()
        logger.info(f"Server status: {status}")
        
        # Run aggregation if enough clients are available
        if len(server.clients) >= args.min_clients:
            result = await server.run_aggregation_round()
            logger.info(f"Aggregation result: {result}")
        
        await asyncio.sleep(60)  # Check every minute


if __name__ == "__main__":
    asyncio.run(main())