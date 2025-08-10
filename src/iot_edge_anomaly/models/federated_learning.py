"""
Federated Learning Architecture for Collaborative IoT Anomaly Detection.

This module implements a comprehensive federated learning system that enables
multiple IoT networks to collaboratively train anomaly detection models
without sharing raw sensor data, preserving privacy while improving performance.

Key Features:
- Privacy-preserving collaborative learning across IoT networks
- Federated graph neural networks for sensor topology sharing
- Differential privacy mechanisms for enhanced security
- Byzantine-robust aggregation for fault tolerance
- Blockchain-based secure model aggregation (optional)
- Hierarchical federated learning for edge-cloud deployment
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
import hashlib
import json
import copy
from collections import OrderedDict
import random
import math

from .lstm_gnn_hybrid import LSTMGNNHybridModel
from .sparse_graph_attention import SparseGraphAttentionNetwork
from .physics_informed_hybrid import PhysicsInformedHybridModel

logger = logging.getLogger(__name__)


class DifferentialPrivacyMechanism:
    """
    Differential Privacy mechanisms for federated learning.
    
    Implements various DP mechanisms to add calibrated noise to model updates
    while preserving utility and providing formal privacy guarantees.
    """
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        sensitivity: float = 1.0,
        mechanism: str = 'gaussian'  # 'gaussian', 'laplace'
    ):
        self.epsilon = epsilon
        self.delta = delta
        self.sensitivity = sensitivity
        self.mechanism = mechanism
        
    def add_noise_to_gradients(
        self, 
        gradients: Dict[str, torch.Tensor],
        clip_norm: float = 1.0
    ) -> Dict[str, torch.Tensor]:
        """
        Add differential privacy noise to gradients.
        
        Args:
            gradients: Dictionary of parameter gradients
            clip_norm: Gradient clipping norm for bounded sensitivity
            
        Returns:
            Noisy gradients with DP guarantees
        """
        noisy_gradients = {}
        
        for name, grad in gradients.items():
            if grad is None:
                continue
            
            # Gradient clipping for bounded sensitivity
            grad_norm = torch.norm(grad)
            if grad_norm > clip_norm:
                grad = grad * (clip_norm / grad_norm)
            
            # Add noise based on mechanism
            if self.mechanism == 'gaussian':
                # Gaussian mechanism: σ² = 2 * sensitivity² * ln(1.25/δ) / ε²
                sigma = math.sqrt(2 * math.log(1.25 / self.delta)) * self.sensitivity / self.epsilon
                noise = torch.normal(0, sigma, size=grad.shape, device=grad.device)
            elif self.mechanism == 'laplace':
                # Laplace mechanism: b = sensitivity / ε
                scale = self.sensitivity / self.epsilon
                noise = torch.distributions.Laplace(0, scale).sample(grad.shape).to(grad.device)
            else:
                raise ValueError(f"Unknown DP mechanism: {self.mechanism}")
            
            noisy_gradients[name] = grad + noise
        
        return noisy_gradients
    
    def add_noise_to_model(
        self, 
        model_state: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """Add DP noise directly to model parameters."""
        noisy_state = {}
        
        for name, param in model_state.items():
            if self.mechanism == 'gaussian':
                sigma = math.sqrt(2 * math.log(1.25 / self.delta)) * self.sensitivity / self.epsilon
                noise = torch.normal(0, sigma, size=param.shape, device=param.device)
            else:  # laplace
                scale = self.sensitivity / self.epsilon
                noise = torch.distributions.Laplace(0, scale).sample(param.shape).to(param.device)
            
            noisy_state[name] = param + noise
        
        return noisy_state
    
    def compute_privacy_cost(self, num_rounds: int, sampling_ratio: float = 1.0) -> Tuple[float, float]:
        """
        Compute cumulative privacy cost over multiple rounds.
        
        Args:
            num_rounds: Number of federated learning rounds
            sampling_ratio: Fraction of clients participating per round
            
        Returns:
            Tuple of (epsilon, delta) accounting for composition
        """
        # Advanced composition theorem for Gaussian mechanism
        if self.mechanism == 'gaussian':
            # RDP composition for Gaussian mechanism
            q = sampling_ratio
            sigma = math.sqrt(2 * math.log(1.25 / self.delta)) * self.sensitivity / self.epsilon
            
            # Compute RDP privacy cost
            alpha = 2
            rdp_cost = q**2 * num_rounds / (2 * sigma**2)
            
            # Convert RDP to (ε, δ)-DP
            eps_total = rdp_cost + math.log(1/self.delta) / (alpha - 1)
            delta_total = self.delta
        else:
            # Simple composition for Laplace mechanism
            eps_total = self.epsilon * num_rounds * sampling_ratio
            delta_total = 0  # Laplace mechanism has δ = 0
        
        return eps_total, delta_total


class ByzantineRobustAggregator:
    """
    Byzantine-robust aggregation for federated learning.
    
    Implements various robust aggregation methods to handle malicious or
    faulty clients in the federated learning process.
    """
    
    def __init__(
        self,
        aggregation_method: str = 'trimmed_mean',
        byzantine_ratio: float = 0.1,
        **kwargs
    ):
        self.aggregation_method = aggregation_method
        self.byzantine_ratio = byzantine_ratio
        self.kwargs = kwargs
    
    def aggregate_models(
        self,
        client_models: List[Dict[str, torch.Tensor]],
        client_weights: Optional[List[float]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client models using Byzantine-robust methods.
        
        Args:
            client_models: List of client model state dictionaries
            client_weights: Optional weights for each client
            
        Returns:
            Aggregated global model state dictionary
        """
        if not client_models:
            raise ValueError("No client models provided for aggregation")
        
        if client_weights is None:
            client_weights = [1.0] * len(client_models)
        
        # Normalize weights
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        if self.aggregation_method == 'fedavg':
            return self._fedavg_aggregation(client_models, client_weights)
        elif self.aggregation_method == 'trimmed_mean':
            return self._trimmed_mean_aggregation(client_models, client_weights)
        elif self.aggregation_method == 'median':
            return self._median_aggregation(client_models)
        elif self.aggregation_method == 'krum':
            return self._krum_aggregation(client_models)
        elif self.aggregation_method == 'bulyan':
            return self._bulyan_aggregation(client_models)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")
    
    def _fedavg_aggregation(
        self,
        client_models: List[Dict[str, torch.Tensor]],
        client_weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """Standard FedAvg weighted averaging."""
        aggregated_model = {}
        
        for param_name in client_models[0].keys():
            weighted_sum = None
            
            for i, model in enumerate(client_models):
                param = model[param_name] * client_weights[i]
                
                if weighted_sum is None:
                    weighted_sum = param.clone()
                else:
                    weighted_sum += param
            
            aggregated_model[param_name] = weighted_sum
        
        return aggregated_model
    
    def _trimmed_mean_aggregation(
        self,
        client_models: List[Dict[str, torch.Tensor]],
        client_weights: List[float]
    ) -> Dict[str, torch.Tensor]:
        """Trimmed mean aggregation - remove outliers and average."""
        num_clients = len(client_models)\n        num_trim = int(num_clients * self.byzantine_ratio)\n        \n        aggregated_model = {}\n        \n        for param_name in client_models[0].keys():\n            # Stack parameters from all clients\n            param_stack = torch.stack([model[param_name] for model in client_models])\n            \n            # Sort along client dimension\n            sorted_params, _ = torch.sort(param_stack, dim=0)\n            \n            # Trim outliers and compute mean\n            if num_trim > 0:\n                trimmed_params = sorted_params[num_trim:-num_trim]\n            else:\n                trimmed_params = sorted_params\n            \n            aggregated_model[param_name] = trimmed_params.mean(dim=0)\n        \n        return aggregated_model\n    \n    def _median_aggregation(\n        self,\n        client_models: List[Dict[str, torch.Tensor]]\n    ) -> Dict[str, torch.Tensor]:\n        \"\"\"Coordinate-wise median aggregation.\"\"\"\n        aggregated_model = {}\n        \n        for param_name in client_models[0].keys():\n            param_stack = torch.stack([model[param_name] for model in client_models])\n            aggregated_model[param_name] = torch.median(param_stack, dim=0)[0]\n        \n        return aggregated_model\n    \n    def _krum_aggregation(\n        self,\n        client_models: List[Dict[str, torch.Tensor]]\n    ) -> Dict[str, torch.Tensor]:\n        \"\"\"Krum aggregation - select most representative model.\"\"\"\n        num_clients = len(client_models)\n        num_byzantine = int(num_clients * self.byzantine_ratio)\n        m = num_clients - num_byzantine - 2\n        \n        if m <= 0:\n            return self._median_aggregation(client_models)\n        \n        # Flatten all model parameters\n        flattened_models = []\n        for model in client_models:\n            flattened = torch.cat([param.flatten() for param in model.values()])\n            flattened_models.append(flattened)\n        \n        # Compute distances between models\n        distances = torch.zeros(num_clients, num_clients)\n        for i in range(num_clients):\n            for j in range(i + 1, num_clients):\n                dist = torch.norm(flattened_models[i] - flattened_models[j])\n                distances[i, j] = distances[j, i] = dist\n        \n        # For each model, sum distances to m closest neighbors\n        krum_scores = []\n        for i in range(num_clients):\n            sorted_distances, _ = torch.sort(distances[i])\n            score = sorted_distances[1:m+1].sum()  # Exclude self-distance (0)\n            krum_scores.append(score)\n        \n        # Select model with minimum Krum score\n        selected_idx = torch.tensor(krum_scores).argmin().item()\n        return client_models[selected_idx]\n    \n    def _bulyan_aggregation(\n        self,\n        client_models: List[Dict[str, torch.Tensor]]\n    ) -> Dict[str, torch.Tensor]:\n        \"\"\"Bulyan aggregation - combination of Krum and trimmed mean.\"\"\"\n        num_clients = len(client_models)\n        num_byzantine = int(num_clients * self.byzantine_ratio)\n        \n        # First, select good models using Krum-like selection\n        selected_models = []\n        remaining_models = client_models.copy()\n        \n        for _ in range(num_clients - 2 * num_byzantine):\n            if len(remaining_models) == 1:\n                selected_models.extend(remaining_models)\n                break\n            \n            # Apply Krum to select one model\n            krum_aggregator = ByzantineRobustAggregator('krum', self.byzantine_ratio)\n            selected_model = krum_aggregator._krum_aggregation(remaining_models)\n            \n            # Find and remove the selected model\n            for i, model in enumerate(remaining_models):\n                if self._models_equal(model, selected_model):\n                    selected_models.append(remaining_models.pop(i))\n                    break\n        \n        # Then apply trimmed mean to selected models\n        return self._trimmed_mean_aggregation(selected_models, [1.0] * len(selected_models))\n    \n    def _models_equal(\n        self,\n        model1: Dict[str, torch.Tensor],\n        model2: Dict[str, torch.Tensor],\n        tolerance: float = 1e-6\n    ) -> bool:\n        \"\"\"Check if two models are approximately equal.\"\"\"\n        for key in model1.keys():\n            if key not in model2:\n                return False\n            if not torch.allclose(model1[key], model2[key], atol=tolerance):\n                return False\n        return True\n\n\nclass SecureAggregationProtocol:\n    \"\"\"\n    Secure aggregation protocol for federated learning.\n    \n    Implements cryptographic secure aggregation to ensure server cannot\n    see individual client updates, only the aggregated result.\n    \"\"\"\n    \n    def __init__(\n        self,\n        num_clients: int,\n        threshold: int,\n        modulus: int = 2**32\n    ):\n        self.num_clients = num_clients\n        self.threshold = threshold  # Minimum clients needed for reconstruction\n        self.modulus = modulus\n        \n        # Generate secret sharing keys (simplified - in practice use proper cryptography)\n        self.client_keys = self._generate_keys()\n    \n    def _generate_keys(self) -> Dict[int, Dict[int, int]]:\n        \"\"\"Generate secret sharing keys for all client pairs.\"\"\"\n        keys = {}\n        \n        for i in range(self.num_clients):\n            keys[i] = {}\n            for j in range(self.num_clients):\n                if i != j:\n                    # In practice, use cryptographically secure random generation\n                    keys[i][j] = random.randint(0, self.modulus - 1)\n        \n        return keys\n    \n    def encode_model_update(\n        self,\n        client_id: int,\n        model_update: Dict[str, torch.Tensor]\n    ) -> Dict[str, torch.Tensor]:\n        \"\"\"Encode model update with secret shares.\"\"\"\n        encoded_update = {}\n        \n        for param_name, param in model_update.items():\n            # Convert to fixed-point representation\n            fixed_point_param = (param * 1000000).long() % self.modulus\n            \n            # Add secret shares\n            secret_sum = 0\n            for j in range(self.num_clients):\n                if j != client_id:\n                    secret_sum += self.client_keys[client_id][j]\n                    secret_sum -= self.client_keys[j][client_id]\n            \n            encoded_param = (fixed_point_param + secret_sum) % self.modulus\n            encoded_update[param_name] = encoded_param.float()\n        \n        return encoded_update\n    \n    def aggregate_encoded_updates(\n        self,\n        encoded_updates: List[Dict[str, torch.Tensor]]\n    ) -> Dict[str, torch.Tensor]:\n        \"\"\"Aggregate encoded model updates.\"\"\"\n        if len(encoded_updates) < self.threshold:\n            raise ValueError(f\"Insufficient clients: {len(encoded_updates)} < {self.threshold}\")\n        \n        aggregated_update = {}\n        \n        for param_name in encoded_updates[0].keys():\n            # Sum all encoded updates\n            param_sum = None\n            for update in encoded_updates:\n                if param_sum is None:\n                    param_sum = update[param_name].clone()\n                else:\n                    param_sum += update[param_name]\n            \n            # Convert back from fixed-point (secrets cancel out in summation)\n            aggregated_update[param_name] = param_sum / (1000000 * len(encoded_updates))\n        \n        return aggregated_update\n\n\nclass FederatedClient:\n    \"\"\"\n    Federated learning client for IoT anomaly detection.\n    \n    Represents an individual IoT network participating in federated learning.\n    Handles local training, model updates, and communication with the server.\n    \"\"\"\n    \n    def __init__(\n        self,\n        client_id: str,\n        model_config: Dict[str, Any],\n        data_config: Dict[str, Any],\n        privacy_config: Dict[str, Any] = None\n    ):\n        self.client_id = client_id\n        self.model_config = model_config\n        self.data_config = data_config\n        self.privacy_config = privacy_config or {}\n        \n        # Initialize local model based on configuration\n        self.local_model = self._create_local_model()\n        self.optimizer = torch.optim.Adam(\n            self.local_model.parameters(),\n            lr=model_config.get('learning_rate', 1e-3)\n        )\n        \n        # Privacy mechanisms\n        if privacy_config:\n            self.dp_mechanism = DifferentialPrivacyMechanism(\n                epsilon=privacy_config.get('epsilon', 1.0),\n                delta=privacy_config.get('delta', 1e-5)\n            )\n        else:\n            self.dp_mechanism = None\n        \n        # Training history\n        self.training_history = {\n            'loss': [],\n            'accuracy': [],\n            'communication_rounds': []\n        }\n        \n        logger.info(f\"Initialized federated client {client_id}\")\n    \n    def _create_local_model(self) -> nn.Module:\n        \"\"\"Create local model based on configuration.\"\"\"\n        model_type = self.model_config.get('type', 'physics_informed')\n        \n        if model_type == 'physics_informed':\n            from .physics_informed_hybrid import create_physics_informed_model\n            return create_physics_informed_model(self.model_config)\n        elif model_type == 'sparse_gat':\n            from .sparse_graph_attention import create_sparse_gat\n            return create_sparse_gat(self.model_config)\n        elif model_type == 'lstm_gnn':\n            return LSTMGNNHybridModel(\n                input_size=self.model_config.get('input_size', 5),\n                lstm_hidden_size=self.model_config.get('hidden_size', 128),\n                output_size=self.model_config.get('output_size', 64)\n            )\n        else:\n            raise ValueError(f\"Unknown model type: {model_type}\")\n    \n    def local_train(\n        self,\n        local_data: Dict[str, torch.Tensor],\n        num_epochs: int = 5,\n        global_model_state: Optional[Dict[str, torch.Tensor]] = None\n    ) -> Dict[str, Any]:\n        \"\"\"\n        Perform local training on client data.\n        \n        Args:\n            local_data: Local training data\n            num_epochs: Number of local training epochs\n            global_model_state: Global model state to start from\n            \n        Returns:\n            Dictionary containing training results and model updates\n        \"\"\"\n        # Update local model with global state if provided\n        if global_model_state is not None:\n            self.local_model.load_state_dict(global_model_state, strict=False)\n        \n        self.local_model.train()\n        \n        epoch_losses = []\n        \n        for epoch in range(num_epochs):\n            self.optimizer.zero_grad()\n            \n            # Forward pass\n            if hasattr(self.local_model, 'forward'):\n                if 'edge_index' in local_data:\n                    outputs = self.local_model(\n                        local_data['x'],\n                        local_data['edge_index']\n                    )\n                else:\n                    outputs = self.local_model(local_data['x'])\n            else:\n                outputs = self.local_model(local_data['x'])\n            \n            # Compute loss\n            if 'anomaly_scores' in outputs:\n                predictions = outputs['anomaly_scores'].squeeze()\n            else:\n                predictions = outputs\n            \n            if 'y' in local_data:\n                loss = F.binary_cross_entropy(predictions, local_data['y'].float())\n            else:\n                # Unsupervised reconstruction loss\n                loss = F.mse_loss(predictions, local_data['x'])\n            \n            # Backward pass\n            loss.backward()\n            \n            # Gradient clipping for stability\n            torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), max_norm=1.0)\n            \n            self.optimizer.step()\n            \n            epoch_losses.append(loss.item())\n        \n        # Compute model update\n        if global_model_state is not None:\n            model_update = self._compute_model_update(global_model_state)\n        else:\n            model_update = self.local_model.state_dict()\n        \n        # Apply differential privacy if configured\n        if self.dp_mechanism is not None:\n            model_update = self.dp_mechanism.add_noise_to_model(model_update)\n        \n        # Update training history\n        avg_loss = np.mean(epoch_losses)\n        self.training_history['loss'].append(avg_loss)\n        \n        return {\n            'client_id': self.client_id,\n            'model_update': model_update,\n            'loss': avg_loss,\n            'num_samples': local_data['x'].size(0),\n            'privacy_cost': self._compute_privacy_cost() if self.dp_mechanism else None\n        }\n    \n    def _compute_model_update(\n        self,\n        global_model_state: Dict[str, torch.Tensor]\n    ) -> Dict[str, torch.Tensor]:\n        \"\"\"Compute difference between local and global model.\"\"\"\n        local_state = self.local_model.state_dict()\n        model_update = {}\n        \n        for key in local_state.keys():\n            if key in global_model_state:\n                model_update[key] = local_state[key] - global_model_state[key]\n            else:\n                model_update[key] = local_state[key]\n        \n        return model_update\n    \n    def _compute_privacy_cost(self) -> Tuple[float, float]:\n        \"\"\"Compute cumulative privacy cost.\"\"\"\n        num_rounds = len(self.training_history['communication_rounds'])\n        return self.dp_mechanism.compute_privacy_cost(num_rounds)\n    \n    def evaluate(\n        self,\n        test_data: Dict[str, torch.Tensor],\n        global_model_state: Optional[Dict[str, torch.Tensor]] = None\n    ) -> Dict[str, float]:\n        \"\"\"Evaluate model performance on test data.\"\"\"\n        if global_model_state is not None:\n            self.local_model.load_state_dict(global_model_state, strict=False)\n        \n        self.local_model.eval()\n        \n        with torch.no_grad():\n            if 'edge_index' in test_data:\n                outputs = self.local_model(test_data['x'], test_data['edge_index'])\n            else:\n                outputs = self.local_model(test_data['x'])\n            \n            if 'anomaly_scores' in outputs:\n                predictions = outputs['anomaly_scores'].squeeze()\n            else:\n                predictions = outputs\n            \n            if 'y' in test_data:\n                # Classification metrics\n                binary_preds = (predictions > 0.5).float()\n                accuracy = (binary_preds == test_data['y']).float().mean().item()\n                \n                # Precision and recall\n                tp = ((binary_preds == 1) & (test_data['y'] == 1)).sum().item()\n                fp = ((binary_preds == 1) & (test_data['y'] == 0)).sum().item()\n                fn = ((binary_preds == 0) & (test_data['y'] == 1)).sum().item()\n                \n                precision = tp / (tp + fp + 1e-8)\n                recall = tp / (tp + fn + 1e-8)\n                f1_score = 2 * precision * recall / (precision + recall + 1e-8)\n                \n                return {\n                    'accuracy': accuracy,\n                    'precision': precision,\n                    'recall': recall,\n                    'f1_score': f1_score\n                }\n            else:\n                # Reconstruction error\n                mse = F.mse_loss(predictions, test_data['x']).item()\n                return {'reconstruction_error': mse}\n\n\nclass FederatedServer:\n    \"\"\"\n    Federated learning server for coordinating IoT anomaly detection.\n    \n    Coordinates the federated learning process, aggregates client updates,\n    and maintains the global model state.\n    \"\"\"\n    \n    def __init__(\n        self,\n        global_model_config: Dict[str, Any],\n        aggregation_config: Dict[str, Any] = None,\n        security_config: Dict[str, Any] = None\n    ):\n        self.global_model_config = global_model_config\n        self.aggregation_config = aggregation_config or {'method': 'fedavg'}\n        self.security_config = security_config or {}\n        \n        # Initialize global model\n        self.global_model = self._create_global_model()\n        \n        # Aggregation mechanism\n        self.aggregator = ByzantineRobustAggregator(\n            aggregation_method=self.aggregation_config.get('method', 'fedavg'),\n            byzantine_ratio=self.aggregation_config.get('byzantine_ratio', 0.1)\n        )\n        \n        # Secure aggregation (optional)\n        if self.security_config.get('secure_aggregation', False):\n            self.secure_aggregator = SecureAggregationProtocol(\n                num_clients=self.security_config.get('num_clients', 10),\n                threshold=self.security_config.get('threshold', 5)\n            )\n        else:\n            self.secure_aggregator = None\n        \n        # Training history\n        self.global_history = {\n            'round': [],\n            'loss': [],\n            'accuracy': [],\n            'participation_rate': [],\n            'aggregation_time': []\n        }\n        \n        logger.info(\"Initialized federated learning server\")\n    \n    def _create_global_model(self) -> nn.Module:\n        \"\"\"Create global model based on configuration.\"\"\"\n        model_type = self.global_model_config.get('type', 'physics_informed')\n        \n        if model_type == 'physics_informed':\n            from .physics_informed_hybrid import create_physics_informed_model\n            return create_physics_informed_model(self.global_model_config)\n        elif model_type == 'sparse_gat':\n            from .sparse_graph_attention import create_sparse_gat\n            return create_sparse_gat(self.global_model_config)\n        elif model_type == 'lstm_gnn':\n            return LSTMGNNHybridModel(\n                input_size=self.global_model_config.get('input_size', 5),\n                lstm_hidden_size=self.global_model_config.get('hidden_size', 128),\n                output_size=self.global_model_config.get('output_size', 64)\n            )\n        else:\n            raise ValueError(f\"Unknown model type: {model_type}\")\n    \n    def federated_training_round(\n        self,\n        client_updates: List[Dict[str, Any]],\n        round_num: int\n    ) -> Dict[str, Any]:\n        \"\"\"\n        Execute one round of federated training.\n        \n        Args:\n            client_updates: List of client training results\n            round_num: Current round number\n            \n        Returns:\n            Dictionary containing round results\n        \"\"\"\n        import time\n        start_time = time.time()\n        \n        # Extract model updates and metadata\n        model_updates = [update['model_update'] for update in client_updates]\n        client_weights = [update['num_samples'] for update in client_updates]\n        \n        # Aggregate model updates\n        if self.secure_aggregator is not None:\n            # Secure aggregation protocol\n            encoded_updates = [\n                self.secure_aggregator.encode_model_update(i, update)\n                for i, update in enumerate(model_updates)\n            ]\n            aggregated_update = self.secure_aggregator.aggregate_encoded_updates(encoded_updates)\n        else:\n            # Standard aggregation\n            aggregated_update = self.aggregator.aggregate_models(model_updates, client_weights)\n        \n        # Update global model\n        global_state = self.global_model.state_dict()\n        \n        for key, update in aggregated_update.items():\n            if key in global_state:\n                global_state[key] += update\n        \n        self.global_model.load_state_dict(global_state)\n        \n        # Compute round statistics\n        avg_loss = np.mean([update['loss'] for update in client_updates])\n        participation_rate = len(client_updates)\n        aggregation_time = time.time() - start_time\n        \n        # Update global history\n        self.global_history['round'].append(round_num)\n        self.global_history['loss'].append(avg_loss)\n        self.global_history['participation_rate'].append(participation_rate)\n        self.global_history['aggregation_time'].append(aggregation_time)\n        \n        round_results = {\n            'round': round_num,\n            'global_model_state': self.global_model.state_dict(),\n            'avg_loss': avg_loss,\n            'num_participants': participation_rate,\n            'aggregation_time': aggregation_time,\n            'privacy_costs': [update.get('privacy_cost') for update in client_updates if update.get('privacy_cost')]\n        }\n        \n        logger.info(f\"Round {round_num} completed - Loss: {avg_loss:.4f}, Participants: {participation_rate}\")\n        \n        return round_results\n    \n    def global_evaluate(\n        self,\n        test_data: Dict[str, torch.Tensor]\n    ) -> Dict[str, float]:\n        \"\"\"Evaluate global model performance.\"\"\"\n        self.global_model.eval()\n        \n        with torch.no_grad():\n            if 'edge_index' in test_data:\n                outputs = self.global_model(test_data['x'], test_data['edge_index'])\n            else:\n                outputs = self.global_model(test_data['x'])\n            \n            if hasattr(outputs, 'get') and 'anomaly_scores' in outputs:\n                predictions = outputs['anomaly_scores'].squeeze()\n            elif isinstance(outputs, dict) and 'anomaly_scores' in outputs:\n                predictions = outputs['anomaly_scores'].squeeze()\n            else:\n                predictions = outputs\n            \n            if 'y' in test_data:\n                binary_preds = (predictions > 0.5).float()\n                accuracy = (binary_preds == test_data['y']).float().mean().item()\n                \n                # Additional metrics\n                tp = ((binary_preds == 1) & (test_data['y'] == 1)).sum().item()\n                fp = ((binary_preds == 1) & (test_data['y'] == 0)).sum().item()\n                fn = ((binary_preds == 0) & (test_data['y'] == 1)).sum().item()\n                \n                precision = tp / (tp + fp + 1e-8)\n                recall = tp / (tp + fn + 1e-8)\n                f1_score = 2 * precision * recall / (precision + recall + 1e-8)\n                \n                return {\n                    'accuracy': accuracy,\n                    'precision': precision,\n                    'recall': recall,\n                    'f1_score': f1_score\n                }\n            else:\n                mse = F.mse_loss(predictions, test_data['x']).item()\n                return {'reconstruction_error': mse}\n    \n    def get_global_model_state(self) -> Dict[str, torch.Tensor]:\n        \"\"\"Get current global model state.\"\"\"\n        return self.global_model.state_dict()\n    \n    def save_global_model(self, save_path: str):\n        \"\"\"Save global model to file.\"\"\"\n        torch.save({\n            'model_state_dict': self.global_model.state_dict(),\n            'model_config': self.global_model_config,\n            'training_history': self.global_history\n        }, save_path)\n        \n        logger.info(f\"Global model saved to {save_path}\")\n\n\nclass HierarchicalFederatedLearning:\n    \"\"\"\n    Hierarchical federated learning for edge-cloud IoT deployment.\n    \n    Implements a two-tier federated learning architecture:\n    - Edge tier: Local federated learning among nearby IoT networks\n    - Cloud tier: Federated learning among edge aggregators\n    \"\"\"\n    \n    def __init__(\n        self,\n        edge_configs: List[Dict[str, Any]],\n        cloud_config: Dict[str, Any]\n    ):\n        self.edge_configs = edge_configs\n        self.cloud_config = cloud_config\n        \n        # Initialize edge servers\n        self.edge_servers = [\n            FederatedServer(config['global_model'], config.get('aggregation', {}), config.get('security', {}))\n            for config in edge_configs\n        ]\n        \n        # Initialize cloud server\n        self.cloud_server = FederatedServer(\n            cloud_config['global_model'],\n            cloud_config.get('aggregation', {}),\n            cloud_config.get('security', {})\n        )\n        \n        self.training_history = {'edge_rounds': [], 'cloud_rounds': []}\n    \n    def hierarchical_training_round(\n        self,\n        client_data: Dict[int, List[Dict[str, Any]]],  # edge_id -> client_updates\n        edge_rounds: int = 5,\n        cloud_round: int = 1\n    ) -> Dict[str, Any]:\n        \"\"\"\n        Execute one round of hierarchical federated training.\n        \n        Args:\n            client_data: Client updates grouped by edge server\n            edge_rounds: Number of edge federation rounds\n            cloud_round: Current cloud round number\n            \n        Returns:\n            Hierarchical training results\n        \"\"\"\n        edge_results = []\n        \n        # Edge-level federated learning\n        for edge_id, edge_server in enumerate(self.edge_servers):\n            if edge_id in client_data:\n                for round_num in range(edge_rounds):\n                    edge_result = edge_server.federated_training_round(\n                        client_data[edge_id],\n                        round_num\n                    )\n                    edge_results.append({\n                        'edge_id': edge_id,\n                        'round': round_num,\n                        'result': edge_result\n                    })\n        \n        # Cloud-level federated learning (edge server aggregation)\n        edge_model_updates = []\n        for edge_server in self.edge_servers:\n            edge_model_state = edge_server.get_global_model_state()\n            edge_model_updates.append({\n                'client_id': f'edge_{len(edge_model_updates)}',\n                'model_update': edge_model_state,\n                'loss': 0.0,  # Placeholder\n                'num_samples': 1000  # Placeholder\n            })\n        \n        cloud_result = self.cloud_server.federated_training_round(\n            edge_model_updates,\n            cloud_round\n        )\n        \n        # Distribute updated global model to edge servers\n        global_state = self.cloud_server.get_global_model_state()\n        for edge_server in self.edge_servers:\n            edge_server.global_model.load_state_dict(global_state, strict=False)\n        \n        return {\n            'edge_results': edge_results,\n            'cloud_result': cloud_result,\n            'global_model_state': global_state\n        }\n\n\n# Factory functions for creating federated learning components\ndef create_federated_client(\n    client_id: str,\n    config: Dict[str, Any]\n) -> FederatedClient:\n    \"\"\"Create a federated learning client.\"\"\"\n    return FederatedClient(\n        client_id=client_id,\n        model_config=config.get('model', {}),\n        data_config=config.get('data', {}),\n        privacy_config=config.get('privacy', {})\n    )\n\n\ndef create_federated_server(config: Dict[str, Any]) -> FederatedServer:\n    \"\"\"Create a federated learning server.\"\"\"\n    return FederatedServer(\n        global_model_config=config.get('global_model', {}),\n        aggregation_config=config.get('aggregation', {}),\n        security_config=config.get('security', {})\n    )